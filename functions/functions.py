from logs.logger import log_evaluation, log_best

import time
import json
import gc

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn import metrics
import logging
from contextlib import contextmanager
import requests


# using ideas from this kernel: https://www.kaggle.com/notslush/altair-visualization-2018-stackoverflow-surve

def fast_auc(y_true, y_prob):
    """
    fast roc_auc computation: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc


def eval_auc(y_true, y_pred):
    """
    Fast auc eval function for lgb.
    """
    return 'auc', fast_auc(y_true, y_pred), True


def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    maes = (y_true-y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()
    

# def train_model_regression(X, X_test, y, params, folds=None, model_type='lgb', eval_metric='mae', columns=None, plot_feature_importance=False, model=None,
#                                verbose=10000, early_stopping_rounds=200, n_estimators=50000, splits=None, n_folds=3):
#     """
#     A function to train a variety of regression models.
#     Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.
#
#     :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
#     :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
#     :params: y - target
#     :params: folds - folds to split data
#     :params: model_type - type of model to use
#     :params: eval_metric - metric to use
#     :params: columns - columns to use. If None - use all columns
#     :params: plot_feature_importance - whether to plot feature importance of LGB
#     :params: model - sklearn model, works only for "sklearn" model type
#
#     """
#     columns = X.columns if columns is None else columns
#     X_test = X_test[columns]
#     splits = folds.split(X) if splits is None else splits
#     n_splits = folds.n_splits if splits is None else n_folds
#
#     # to set up scoring parameters
#     metrics_dict = {'mae': {'lgb_metric_name': 'mae',
#                             'catboost_metric_name': 'MAE',
#                             'sklearn_scoring_function': metrics.mean_absolute_error},
#                     'group_mae': {'lgb_metric_name': 'mae',
#                                   'catboost_metric_name': 'MAE',
#                                   'scoring_function': group_mean_log_mae},
#                     'mse': {'lgb_metric_name': 'mse',
#                             'catboost_metric_name': 'MSE',
#                             'sklearn_scoring_function': metrics.mean_squared_error}
#                     }
#
#
#     result_dict = {}
#
#     # out-of-fold predictions on train data
#     oof = np.zeros(len(X))
#
#     # averaged predictions on train data
#     prediction = np.zeros(len(X_test))
#
#     # list of scores on folds
#     scores = []
#     feature_importance = pd.DataFrame()
#
#     # split and train on folds
#     for fold_n, (train_index, valid_index) in enumerate(splits):
#         if verbose:
#             print(f'Fold {fold_n + 1} started at {time.ctime()}')
#         if type(X) == np.ndarray:
#             X_train, X_valid = X[columns][train_index], X[columns][valid_index]
#             y_train, y_valid = y[train_index], y[valid_index]
#         else:
#             X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
#             y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
#
#         if model_type == 'lgb':
#             model = lgb.LGBMRegressor(**params, n_estimators=n_estimators, n_jobs=-1)
#             model.fit(X_train, y_train,
#                     eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
#                     verbose=verbose, early_stopping_rounds=early_stopping_rounds)
#
#             y_pred_valid = model.predict(X_valid)
#             y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
#
#         if model_type == 'xgb':
#             train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
#             valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)
#
#             watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
#             model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=verbose, params=params)
#             y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
#             y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
#
#         if model_type == 'sklearn':
#             model = model
#             model.fit(X_train, y_train)
#
#             y_pred_valid = model.predict(X_valid).reshape(-1,)
#             score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)
#             print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')
#             print('')
#
#             y_pred = model.predict(X_test).reshape(-1,)
#
#         if model_type == 'cat':
#             model = CatBoostRegressor(iterations=20000,  eval_metric=metrics_dict[eval_metric]['catboost_metric_name'], **params,
#                                       loss_function=metrics_dict[eval_metric]['catboost_metric_name'])
#             model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)
#
#             y_pred_valid = model.predict(X_valid)
#             y_pred = model.predict(X_test)
#
#         oof[valid_index] = y_pred_valid.reshape(-1,)
#         if eval_metric != 'group_mae':
#             scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))
#         else:
#             scores.append(metrics_dict[eval_metric]['scoring_function'](y_valid, y_pred_valid, X_valid['type']))
#
#         prediction += y_pred
#
#         if model_type == 'lgb' and plot_feature_importance:
#             # feature importance
#             fold_importance = pd.DataFrame()
#             fold_importance["feature"] = columns
#             fold_importance["importance"] = model.feature_importances_
#             fold_importance["fold"] = fold_n + 1
#             feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
#
#     prediction /= n_splits
#     print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
#
#     result_dict['oof'] = oof
#     result_dict['prediction'] = prediction
#     result_dict['scores'] = scores
#
#     if model_type == 'lgb':
#         if plot_feature_importance:
#             feature_importance["importance"] /= n_splits
#             cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
#                 by="importance", ascending=False)[:50].index
#
#             best_features = feature_importance.loc[feature_importance.feature.isin(cols)]
#
#             plt.figure(figsize=(16, 12));
#             sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
#             plt.title('LGB Features (avg over folds)');
#
#             result_dict['feature_importance'] = feature_importance
#
#     return result_dict
#


# def train_model_classification(X, X_test, y, params, folds, model_type='lgb',
#                                eval_metric='auc', columns=None, seed=0,
#                                plot_feature_importance=False, model=None,
#                                verbose=10000, early_stopping_rounds=200,
#                                n_estimators=50000, splits=None, n_folds=3,
#                                averaging='usual', n_jobs=-1, cat_cols=""):
#     """
#     A function to train a variety of classification models.
#     Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.
#
#     :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
#     :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
#     :params: y - target
#     :params: folds - folds to split data
#     :params: model_type - type of model to use
#     :params: eval_metric - metric to use
#     :params: columns - columns to use. If None - use all columns
#     :params: plot_feature_importance - whether to plot feature importance of LGB
#     :params: model - sklearn model, works only for "sklearn" model type
#
#     """
#     columns = X.columns if columns is None else columns
#     n_splits = folds.n_splits if splits is None else n_folds
#     X_test = X_test[columns]
#
#     # to set up scoring parameters
#     metrics_dict = {'auc': {'lgb_metric_name': eval_auc,
#                             'catboost_metric_name': 'AUC',
#                             'sklearn_scoring_function': metrics.roc_auc_score},
#                     }
#
#     result_dict = {}
#     if averaging == 'usual':
#         # out-of-fold predictions on train data
#         oof = np.zeros((len(X), 1))
#
#         # averaged predictions on train data
#         prediction = np.zeros((len(X_test), 1))
#
#     elif averaging == 'rank':
#         # out-of-fold predictions on train data
#         oof = np.zeros((len(X), 1))
#
#         # averaged predictions on train data
#         prediction = np.zeros((len(X_test), 1))
#
#
#     # list of scores on folds
#     scores = []
#     feature_importance = pd.DataFrame()
#
#     # split and train on folds
#     for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
#         print(f'\nFold {fold_n + 1} started at {time.ctime()}')
#         if type(X) == np.ndarray:
#             X_train, X_valid = X[columns][train_index], X[columns][valid_index]
#             y_train, y_valid = y[train_index], y[valid_index]
#         else:
#             X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
#             y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
#
#
#         ###### models ##################################################
#         if model_type == 'lgb':
#             # logger
#             logging.debug('\n\n=== lgb training =========')
#             logger = logging.getLogger('main')
#             callbacks = [log_evaluation(logger, period=500)]
#
#             # modelの構築
#             model = lgb.LGBMClassifier(**params, n_estimators=n_estimators, n_jobs=n_jobs, seed=seed)
#             model.fit(X_train, y_train,
#                       eval_set=[(X_train, y_train), (X_valid, y_valid)],
#                       eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
#                       verbose=verbose, early_stopping_rounds=early_stopping_rounds,
#                       categorical_feature=cat_cols, callbacks=callbacks)
#
#             y_pred_valid = model.predict_proba(X_valid)[:, 1]
#             y_pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)[:, 1]
#
#             # best score
#             log_best(model)
#
#         if model_type == 'xgb':
#             train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
#             valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)
#
#             watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
#             model = xgb.train(dtrain=train_data, num_boost_round=n_estimators, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose, params=params)
#             y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
#             y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
#
#         if model_type == 'sklearn':
#             model = model
#             model.fit(X_train, y_train)
#
#             y_pred_valid = model.predict(X_valid).reshape(-1,)
#             score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)
#             print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')
#             print('')
#
#             y_pred = model.predict_proba(X_test)
#
#         if model_type == 'cat':
#             model = CatBoostClassifier(iterations=n_estimators, eval_metric=metrics_dict[eval_metric]['catboost_metric_name'], **params,
#                                       loss_function=metrics_dict[eval_metric]['catboost_metric_name'])
#             model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)
#
#             y_pred_valid = model.predict(X_valid)
#             y_pred = model.predict(X_test)
#
#
#         ##### how to metric ###################################################
#         if averaging == 'usual':
#
#             oof[valid_index] = y_pred_valid.reshape(-1, 1)
#             scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))
#
#             prediction += y_pred.reshape(-1, 1)
#
#         elif averaging == 'rank':
#
#             oof[valid_index] = y_pred_valid.reshape(-1, 1)
#             scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))
#
#             prediction += pd.Series(y_pred).rank().values.reshape(-1, 1)
#
#         if model_type == 'lgb' and plot_feature_importance:
#             # feature importance
#             fold_importance = pd.DataFrame()
#             fold_importance["feature"] = columns
#             fold_importance["importance"] = model.feature_importances_
#             fold_importance["fold"] = fold_n + 1
#             feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
#
#         gc.collect()
#     ####################################################################
#     prediction /= n_splits
#
#     print('\nCV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
#
#     result_dict['oof'] = oof
#     result_dict['prediction'] = prediction
#     result_dict['scores'] = scores
#
#     if model_type == 'lgb':
#         if plot_feature_importance:
#             feature_importance["importance"] /= n_splits
#             cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
#                 by="importance", ascending=False)[:50].index
#
#             best_features = feature_importance.loc[feature_importance.feature.isin(cols)]
#
# #             plt.figure(figsize=(16, 12));
# #             sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
# #             plt.title('LGB Features (avg over folds)');
#
#             result_dict['feature_importance'] = feature_importance
#             result_dict['top_columns'] = cols
#
#     return result_dict



###########################################################################
# reference:https://github.com/amaotone/spica/blob/master/spica/utils.py



def send_line_notification(message, config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    line_token = config['line_token']  # 終わったら無効化する
    line_notify_api = 'https://notify-api.line.me/api/notify'
    message = "\n{}".format(message)
    payload = {'message': message}
    headers = {'Authorization': 'Bearer {}'.format(line_token)}  # 発行したトークン
    requests.post(line_notify_api, data=payload, headers=headers)


@contextmanager
def timer(name, logger=None, level=logging.DEBUG):
    print_ = print if logger is None else lambda msg: logger.log(level, msg)
    t0 = time.time()
    print_(f'[{name}] start')
    yield
    print_(f'[{name}] done in {time.time() - t0:.0f} s')


def timestamp():
    return time.strftime('%y%m%d_%H%M%S')

############################################################################
# reference: https://github.com/upura/ml-competition-template-titanic/blob/master/utils/__init__.py


def load_datasets(feats):
    dfs = [pd.read_feather(f'/tmp/working/IEEE_Fraud_Detection/features/{f}_train.feather') for f in feats]
    X_train = pd.concat(dfs, axis=1)
    dfs = [pd.read_feather(f'/tmp/working/IEEE_Fraud_Detection/features/{f}_test.feather') for f in feats]
    X_test = pd.concat(dfs, axis=1)
    return X_train, X_test


def load_target(target_name):
    train = pd.read_csv('./data/input/train.csv')
    y_train = train[target_name]
    return y_train


############################################################################
# reference: https://www.kaggle.com/kyakovlev/ieee-data-minification



# def reduce_mem_usage(df, verbose=True):
#     numerics: List[str] = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
#     start_mem = df.memory_usage().sum() / 1024**2
#     for col in df.columns:
#         col_type = df[col].dtypes
#         if col_type in numerics:
#             c_min = df[col].min()
#             c_max = df[col].max()
#             if str(col_type)[:3] == 'int':
#                 if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
#                     df[col] = df[col].astype(np.int8)
#                 elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
#                     df[col] = df[col].astype(np.int16)
#                 elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
#                     df[col] = df[col].astype(np.int32)
#                 elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
#                     df[col] = df[col].astype(np.int64)
#             else:
#                 if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
#                     df[col] = df[col].astype(np.float16)
#                 elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
#                     df[col] = df[col].astype(np.float32)
#                 else:
#                     df[col] = df[col].astype(np.float64)
#     end_mem = df.memory_usage().sum() / 1024**2
#     if verbose:
#         print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem,
#                                                                               100 * (start_mem - end_mem) / start_mem))
#     return df



def reduce_mem_usage(df, deep=True, verbose=True, categories=True):
    def memory_usage_mb(df, *args, **kwargs):
        """Dataframe memory usage in MB. """
        return df.memory_usage(*args, **kwargs).sum() / 1024 ** 2
    # All types that we want to change for "lighter" ones.
    # int8 and float16 are not include because we cannot reduce
    # those data types.
    # float32 is not include because float16 has too low precision.
    numeric2reduce = ["int16", "int32", "int64", "float64"]
    start_mem = 0
    if verbose:
        start_mem = memory_usage_mb(df, deep=deep)

    for col, col_type in df.dtypes.iteritems():
        best_type = None
        if col_type == "object":
            df[col] = df[col].astype("category")
            best_type = "category"
        elif col_type in numeric2reduce:
            downcast = "integer" if "int" in str(col_type) else "float"
            df[col] = pd.to_numeric(df[col], downcast=downcast)
            best_type = df[col].dtype.name
        # Log the conversion performed.
        # if verbose and best_type is not None and best_type != str(col_type):
        #     print(f"Column '{col}' converted from {col_type} to {best_type}")

    if verbose:
        end_mem = memory_usage_mb(df, deep=deep)
        diff_mem = start_mem - end_mem
        percent_mem = 100 * diff_mem / start_mem
        print(f"Memory usage decreased from"
              f" {start_mem:.2f}MB to {end_mem:.2f}MB"
              f" ({diff_mem:.2f}MB, {percent_mem:.2f}% reduction)")

    return df

############################################################################
# reference: https://www.kaggle.com/nroman/eda-for-cis-fraud-detection


def relax_data(df_train, df_test, col):
    cv1 = pd.DataFrame(df_train[col].value_counts().reset_index().rename({col: 'train'}, axis=1))
    cv2 = pd.DataFrame(df_test[col].value_counts().reset_index().rename({col: 'test'}, axis=1))
    cv3 = pd.merge(cv1, cv2, on='index', how='outer')
    factor = len(df_test)/len(df_train)
    cv3['train'].fillna(0, inplace=True)
    cv3['test'].fillna(0, inplace=True)
    cv3['remove'] = False
    cv3['remove'] = cv3['remove'] | (cv3['train'] < len(df_train)/10000)
    cv3['remove'] = cv3['remove'] | (factor*cv3['train'] < cv3['test']/3)
    cv3['remove'] = cv3['remove'] | (factor*cv3['train'] > 3*cv3['test'])
    cv3['new'] = cv3.apply(lambda x: x['index'] if x['remove'] == False else 0, axis=1)
    cv3['new'], _ = cv3['new'].factorize(sort=True)
    cv3.set_index('index', inplace=True)
    cc = cv3['new'].to_dict()
    df_train[col] = df_train[col].map(cc)
    df_test[col] = df_test[col].map(cc)

    return df_train, df_test
