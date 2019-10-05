import sys
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import math
from functions.functions import load_datasets, reduce_mem_usage, relax_data
from models.models import train_model_classification
import time
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, TimeSeriesSplit, GroupKFold
import argparse
import json
import logging
import gc
import warnings

# calculate execution time
t1 = time.time()

sys.path.append('/tmp/working/IEEE_Fraud_Detection/')

print("\n'drop_features' is running...")

warnings.filterwarnings("ignore")

# JSONファイルからのconfigの読み込み
parser: ArgumentParser = argparse.ArgumentParser()  # パーサのインスタンス化
parser.add_argument('--config', default='./configs/default.json')  # 受け取る引数の追加。引数は--config, デフォルトも設定
options = parser.parse_args()  # 解析した引数をoptionsに格納
config = json.load(open(options.config))  # 引数に渡したJSONファイルを開く

##### logging #####
now = datetime.datetime.now()
logging.basicConfig(
    filename='./logs/drop_feature_log_{0:%Y%m%d%H%M}.log'.format(now), level=logging.DEBUG
)
logging.debug('./logs/drop_feature_log_{0:%Y%m%d%H%M}.log'.format(now))

##### whether use pickle data or not #####
logging.debug('\n\n=== used feature mode =========')
feature_mode = config['feature_mode']
logging.debug(feature_mode)


# if feature_mode == 'pickle':
#     train = pd.read_pickle('train_2.pkl')
#     test = pd.read_pickle('test_2.pkl')
#     train = train.drop(['uid', 'uid2', 'uid3', 'uid4', 'uid5', 'bank_type'], axis=1)
#     test = test.drop(['uid', 'uid2', 'uid3', 'uid4', 'uid5', 'bank_type'], axis=1)
#
#     # add feature
#     logging.debug('\n\n=== stacking =========')
#     stacking = config['stacking']
#     logging.debug(stacking)
#     if stacking == 'true':
#         train_add = pd.read_csv('nn_oof_train.csv')
#         test_add = pd.read_csv('nn_oof_test.csv')
#         train['oof'] = train_add['oof']
#         test['oof'] = test_add['isFraud']



logging.debug('\n\n=== features =========')
feats = ['transaction_identity_merged', 'd_columns_engineering', 'group_v_pca', 'date_of_month']
logging.debug(feats)

# featherからデータの読み込み
train, test = load_datasets(feats)
# Noneをnp.nanに戻す
train.replace(to_replace=[None], value=np.nan, inplace=True)
test.replace(to_replace=[None], value=np.nan, inplace=True)


cols_to_drop=[]
cols_to_drop += ['V' + str(i) for i in range(1, 340)]
cols_to_drop = list(set(cols_to_drop))
cols_to_drop = list(set(cols_to_drop) & set(list(train)))
train = train.drop(cols_to_drop, axis=1)
test = test.drop(cols_to_drop, axis=1)

print('\nTo drop column : ', cols_to_drop)
logging.debug('\n\n=== Dropped columns =========')
logging.debug(cols_to_drop)

# カテゴリカル変数をintへ
cat_cols = ['id_12',
            'id_13',
            'id_14',
            'id_15',
            'id_16',
            'id_17',
            'id_18',
            'id_19',
            'id_20',
            'id_21',
            'id_22',
            'id_23',
            'id_24',
            'id_25',
            'id_26',
            'id_27',
            'id_28',
            'id_29',
            'id_30',
            'id_31',
            'id_32',
            'id_33',
            'id_34',
            'id_35',
            'id_36',
            'id_37',
            'id_38',
            'DeviceType',
            'DeviceInfo',
            'ProductCD',
            'card4',
            'card6',
            'M4',
            'P_emaildomain',
            'R_emaildomain',
            'card1',
            'card2',
            'card3',
            'card5',
            'addr1',
            'addr2',
            'M1',
            'M2',
            'M3',
            'M5',
            'M6',
            'M7',
            'M8',
            'M9'
            ]


########################### Transform Heavy Dominated columns
total_items = len(train)
keep_cols = ['isFraud']


for col in cat_cols:
    if col in train.columns:
        # train[col] = train[col].fillna('unseen_before_label')
        # test[col] = test[col].fillna('unseen_before_label')
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        le = LabelEncoder()
        le.fit(list(train[col]) + list(test[col]))
        train[col] = le.transform(list(train[col]))
        test[col] = le.transform(list(test[col]))




# Training params
model_type = config['model_type']
logging.debug('\n\n=== model type =========')
logging.debug(model_type)

params = config['{}_params'.format(model_type)]
logging.debug('\n\n=== params =========')
logging.debug(params)

logging.debug('\n\n=== Down sampling Rate =========')
dwsp_rate = config['downsampling_rate']
logging.debug(dwsp_rate)

logging.debug('\n\n=== Bagging times =========')
bag_times = config['bagging_times']
logging.debug(bag_times)

logging.debug('\n\n=== random_seed_average times =========')
random_seed_average_times = config['random_seed_average_times']
logging.debug(random_seed_average_times)

logging.debug('\n\n=== N Folds =========')
n_fold = config['n_fold']
logging.debug(n_fold)

logging.debug('\n\n=== Folds Type =========')
folds_type = {'time_series': TimeSeriesSplit(n_fold), 'k_fold': KFold(n_fold),
              'group_k_fold': GroupKFold(n_fold), 'train_test_split_time_series': 'train_test_split_time_series'}
folds = folds_type[config['folds_type']]
logging.debug(config['folds_type'])
if config['folds_type'] == 'group_k_fold':
    split_groups = train['DT_M']
else:
    split_groups = None


logging.debug('\n\n=== train shape =========')
logging.debug(train.shape)
print('train shape', train.shape)


# Bagging List
y_preds = []
score_list = []
oof_list = []

X_test = test.sort_values('TransactionDT').drop(['TransactionDT', 'TransactionID'], axis=1)
test = test[["TransactionDT", 'TransactionID']]
train_fr = train.loc[train['isFraud'] == 1]
train_nofr = train.loc[train['isFraud'] == 0]
del train
train_nofr_dwsp = train_nofr.sample(n=math.floor(train_nofr.shape[0] * dwsp_rate), replace=True, random_state=0)
train_dwsp = pd.concat([train_fr, train_nofr_dwsp]).sample(frac=1, random_state=0)
X = train_dwsp.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
y = train_dwsp.sort_values('TransactionDT')['isFraud']
del train_dwsp
gc.collect()
print('\n Down sampling is completed.\n')

save_df, save_df_dummy = load_datasets(['d_columns_engineering'])
retain_cols = list(save_df)
del save_df, save_df_dummy

result_dict_lgb = train_model_classification(
        X=X,
        X_test=X_test,
        y=y,
        params=params,
        folds=folds,
        model_type=model_type,
        eval_metric='auc',
        plot_feature_importance=True,
        verbose=100,
        early_stopping_rounds=10,
        n_estimators=10000,
        averaging='usual',
        n_jobs=-1,
        cat_cols=cat_cols,
        seed=0,
        groups=split_groups
    )


best_score = np.mean(result_dict_lgb['scores'])
drop_cols = []

for col_to_test in retain_cols:
    retain_list = [retain for retain in retain_cols if retain not in drop_cols]
    retain_list.remove(col_to_test)
    print(retain_list)

    X_train = X.drop(col_to_test, axis=1)
    X_test_test = X_test.drop(col_to_test, axis=1)

    print(
        '---------- iter {} start -------------------------------------'
        '------------------------------------------------------------'.format(col_to_test))
    result_dict_lgb = train_model_classification(
        X=X_train,
        X_test=X_test_test,
        y=y,
        params=params,
        folds=folds,
        model_type=model_type,
        eval_metric='auc',
        plot_feature_importance=True,
        verbose=100,
        early_stopping_rounds=10,
        n_estimators=10000,
        averaging='usual',
        n_jobs=-1,
        cat_cols=cat_cols,
        seed=0,
        groups=split_groups
    )

    local_score = np.mean(result_dict_lgb['scores'])

    if local_score < best_score:
        print("performance reduced when", col_to_test, "dropped to", local_score, "from", best_score)
    else:
        drop_cols.append(col_to_test)
        print("performance increased when", col_to_test, "dropped to", local_score, "from", best_score)
    gc.collect()

    print(drop_cols)

print(
    '--------------------------------------------------------------')

print('drop_cols', '\n', drop_cols)
logging.debug('\n\n=== drop cols =========')
logging.debug(drop_cols)

# calculate exectime
t2 = time.time()
exec_time = math.floor((t2 - t1) / 60)
print('execution time: {} min'.format(str(exec_time)), '\n')
logging.debug('\n\n=== Execution Time =========')
logging.debug('{} min'.format(str(exec_time)))
