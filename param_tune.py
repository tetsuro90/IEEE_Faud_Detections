import sys
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import math
from functions.functions import load_datasets
import time
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, TimeSeriesSplit
import argparse
import json
import logging
import gc
import warnings
import optuna
from optuna.pruners import MedianPruner

import lightgbm as lgb
from sklearn import metrics

# calculate execution time
t1 = time.time()

sys.path.append('/tmp/working/IEEE_Fraud_Detection/')

print("\n'param_tune.py' is running...")

warnings.filterwarnings("ignore")

# JSONファイルからのconfigの読み込み
parser: ArgumentParser = argparse.ArgumentParser()  # パーサのインスタンス化
parser.add_argument('--config', default='./configs/default.json')  # 受け取る引数の追加。引数は--config, デフォルトも設定
options = parser.parse_args()  # 解析した引数をoptionsに格納
config = json.load(open(options.config))  # 引数に渡したJSONファイルを開く

##### logging #####
now = datetime.datetime.now()
logging.basicConfig(
    filename='./logs/param_tune_log_{0:%Y%m%d%H%M}.log'.format(now), level=logging.DEBUG
)
logging.debug('./logs/param_tune_log_{0:%Y%m%d%H%M}.log'.format(now))


# Data Load
train = pd.read_pickle('train_2.pkl')
test = pd.read_pickle('test_2.pkl')
train = train.drop(['uid', 'uid2', 'uid3', 'uid4', 'uid5', 'bank_type'], axis=1)
test = test.drop(['uid', 'uid2', 'uid3', 'uid4', 'uid5', 'bank_type'], axis=1)


# Training params
model_type = config['model_type']
logging.debug('\n\n=== model type =========')
logging.debug(model_type)


cat_cols = config['cat_cols']
logging.debug('\n\n=== categorycal features =========')
logging.debug(cat_cols)


X = train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
y = train.sort_values('TransactionDT')['isFraud']

gc.collect()



def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 200, 800),
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 600),
        'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-3, 1e2),
        'objective': trial.suggest_categorical('objective', ["binary"]),
        'max_depth': trial.suggest_categorical('max_depth', [-1]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.01]),
        "boosting_type": trial.suggest_categorical('boosting_type', ["gbdt"]),
        "subsample_freq": trial.suggest_categorical('subsample_freq', [2]),
        'subsample': trial.suggest_uniform('subsample', 0.4, 0.75),
        "bagging_seed": 11,
        "metric": trial.suggest_categorical('metric', ['auc']),
        "verbosity": -1,
        'reg_alpha': trial.suggest_uniform('reg_alpha', 0.5, 5),
        'reg_lambda': trial.suggest_uniform('reg_lambda', 0.5, 5),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.4, 0.75),
        'max_bin': trial.suggest_int("max_bin", 255, 512),
    }
    columns = X.columns
    valid_rate = 0.8
    train_index = np.arange(math.floor(X.shape[0]*valid_rate))
    valid_index = np.arange(math.floor(X.shape[0]*valid_rate), X.shape[0])
    if type(X) == np.ndarray:
        X_train, X_valid = X[columns][train_index], X[columns][valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
    else:
        X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=cat_cols)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train, categorical_feature=cat_cols)

    model = lgb.train(
        params, lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        verbose_eval=500,
        num_boost_round=5000,
        early_stopping_rounds=50
    )
    y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)
    score = metrics.roc_auc_score(y_valid, y_pred_valid)
    return 1 - score  # returnする値を '最小化'するアルゴリズム

print(
    '---------- param tune start -------------------------------------'
    '------------------------------------------------------------')
study = optuna.create_study(sampler=optuna.samplers.TPESampler(),
                            pruner=MedianPruner())
study.optimize(objective, timeout=10800)
best_param = study.best_params
best_score = 1 - study.best_value
print('\nBest Parameters: ', best_param)
print('\nBest Score: ', best_score)

print(
    '--------------------------------------------------------------'
    '------------------------------------------------------------------')


logging.debug('\n\n=== Best Params =========')
logging.debug(best_param)
logging.debug('\n\n=== Best Score =========')
logging.debug(best_score)


print("\n'param_tune.py' completed.")

# calculate exectime
t2 = time.time()
exec_time = math.floor((t2 - t1)/60)
print('execution time: {} min'.format(str(exec_time)), '\n')
logging.debug('\n\n=== Execution Time =========')
logging.debug('{} min'.format(str(exec_time)))
