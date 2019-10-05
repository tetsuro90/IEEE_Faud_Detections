import math
from logs.logger import log_evaluation, log_best
import time
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn import metrics
import logging


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


def train_model_classification(X, X_test, y, params, folds, model_type='lgb',
                               eval_metric='auc', columns=None, seed=0,
                               plot_feature_importance=False, model=None,
                               verbose=10000, early_stopping_rounds=200,
                               n_estimators=50000, splits=None, n_folds=3,
                               averaging='usual', n_jobs=-1, cat_cols="",
                               valid_rate=0.8, groups=None):
    columns = X.columns if columns is None else columns
    n_splits = folds.n_splits if splits is None else n_folds
    X_test = X_test[columns]

    # to set up scoring parameters
    metrics_dict = {'auc': {'lgb_metric_name': eval_auc,
                            'catboost_metric_name': 'AUC',
                            'sklearn_scoring_function': metrics.roc_auc_score},
                    }

    result_dict = {}
    if averaging == 'usual':
        # out-of-fold predictions on train data
        oof = np.zeros((len(X), 1))

        # averaged predictions on train data
        prediction = np.zeros((len(X_test), 1))

    elif averaging == 'rank':
        # out-of-fold predictions on train data
        oof = np.zeros((len(X), 1))

        # averaged predictions on train data
        prediction = np.zeros((len(X_test), 1))

    # list of scores on folds
    scores = []
    feature_importance = pd.DataFrame()

    if folds == 'train_test_split_time_series':
        n_splits = 1
        valid_rate = valid_rate
        train_index = np.arange(math.floor(X.shape[0] * valid_rate))
        valid_index = np.arange(math.floor(X.shape[0] * valid_rate))
        print(f'\ntrain started at {time.ctime()}')
        if type(X) == np.ndarray:
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        ###### models ##################################################
        if model_type == 'lgb':
            # logger
            logging.debug('\n\n=== lgb training =========')
            logger = logging.getLogger('main')
            callbacks = [log_evaluation(logger, period=500)]

            # modelの構築
            model = lgb.LGBMClassifier(**params, n_estimators=n_estimators, n_jobs=n_jobs, seed=seed)
            model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_valid, y_valid)],
                      eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                      verbose=verbose, early_stopping_rounds=early_stopping_rounds,
                      categorical_feature=cat_cols, callbacks=callbacks)

            y_pred_valid = model.predict_proba(X_valid)[:, 1]
            y_pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)[:, 1]

            # best score
            log_best(model)

        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=n_estimators, evals=watchlist,
                              early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns),
                                         ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid).reshape(-1, )
            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)
            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')
            print('')

            y_pred = model.predict_proba(X_test)

        if model_type == 'cat':
            model = CatBoostClassifier(iterations=n_estimators,
                                       eval_metric=metrics_dict[eval_metric]['catboost_metric_name'],
                                       **params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=cat_cols, use_best_model=True,
                      verbose=verbose)

            y_pred_valid = model.predict_proba(X_valid)[:, 1]
            y_pred = model.predict_proba(X_test)[:, 1]

        ##### how to metric ###################################################
        if averaging == 'usual':
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))
            prediction += y_pred.reshape(-1, 1)

        elif averaging == 'rank':
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))
            prediction += pd.Series(y_pred).rank().values.reshape(-1, 1)

        if model_type == 'lgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

        gc.collect()

        prediction /= n_splits
        print('\nCV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

        result_dict['prediction'] = prediction
        result_dict['scores'] = scores

    else:
        # split and train on folds
        for fold_n, (train_index, valid_index) in enumerate(folds.split(X, groups=groups)):
            print(f'\nFold {fold_n + 1} started at {time.ctime()}')
            if type(X) == np.ndarray:
                X_train, X_valid = X[columns][train_index], X[columns][valid_index]
                y_train, y_valid = y[train_index], y[valid_index]
            else:
                X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
                y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

            ###### models ##################################################
            if model_type == 'lgb':
                # logger
                logging.debug('\n\n=== lgb training =========')
                logger = logging.getLogger('main')
                callbacks = [log_evaluation(logger, period=500)]

                # modelの構築
                model = lgb.LGBMClassifier(**params, n_estimators=n_estimators, n_jobs=n_jobs, seed=seed)
                model.fit(X_train, y_train,
                          eval_set=[(X_train, y_train), (X_valid, y_valid)],
                          eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                          verbose=verbose, early_stopping_rounds=early_stopping_rounds,
                          categorical_feature=cat_cols, callbacks=callbacks)

                y_pred_valid = model.predict_proba(X_valid)[:, 1]
                y_pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)[:, 1]

                # best score
                log_best(model)

            if model_type == 'xgb':
                train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
                valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

                watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
                model = xgb.train(dtrain=train_data, num_boost_round=n_estimators, evals=watchlist,
                                  early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose, params=params)
                y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns),
                                             ntree_limit=model.best_ntree_limit)
                y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

            if model_type == 'sklearn':
                model = model
                model.fit(X_train, y_train)

                y_pred_valid = model.predict(X_valid).reshape(-1, )
                score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)
                print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')
                print('')

                y_pred = model.predict_proba(X_test)

            if model_type == 'cat':
                model = CatBoostClassifier(iterations=n_estimators,
                                           eval_metric=metrics_dict[eval_metric]['catboost_metric_name'],
                                           **params)
                model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=cat_cols, use_best_model=True,
                          verbose=verbose)

                y_pred_valid = model.predict_proba(X_valid)[:, 1]
                y_pred = model.predict_proba(X_test)[:, 1]

            gc.collect()

            ##### how to metric ###################################################
            if averaging == 'usual':

                oof[valid_index] = y_pred_valid.reshape(-1, 1)
                scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))

                prediction += y_pred.reshape(-1, 1)

            elif averaging == 'rank':

                oof[valid_index] = y_pred_valid.reshape(-1, 1)
                scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))

                prediction += pd.Series(y_pred).rank().values.reshape(-1, 1)

            if model_type == 'lgb' and plot_feature_importance:
                # feature importance
                fold_importance = pd.DataFrame()
                fold_importance["feature"] = columns
                fold_importance["importance"] = model.feature_importances_
                fold_importance["fold"] = fold_n + 1
                feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

            gc.collect()


        prediction /= n_splits
        print('\nCV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

        result_dict['oof'] = oof
        result_dict['prediction'] = prediction
        result_dict['scores'] = scores
    ####################################################################

    if model_type == 'lgb':
        if plot_feature_importance:
            feature_importance["importance"] /= n_splits
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            #             plt.figure(figsize=(16, 12));
            #             sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            #             plt.title('LGB Features (avg over folds)');

            result_dict['feature_importance'] = feature_importance
            result_dict['top_columns'] = cols

    return result_dict
