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

print("\n'run.py' is running...")

warnings.filterwarnings("ignore")

# JSONファイルからのconfigの読み込み
parser: ArgumentParser = argparse.ArgumentParser()  # パーサのインスタンス化
parser.add_argument('--config', default='./configs/default.json')  # 受け取る引数の追加。引数は--config, デフォルトも設定
options = parser.parse_args()  # 解析した引数をoptionsに格納
config = json.load(open(options.config))  # 引数に渡したJSONファイルを開く

##### logging #####
now = datetime.datetime.now()
logging.basicConfig(
    filename='./logs/log_{0:%Y%m%d%H%M}.log'.format(now), level=logging.DEBUG
)
logging.debug('./logs/log_{0:%Y%m%d%H%M}.log'.format(now))

# submission file
folder_path = '/tmp/working/IEEE_Fraud_Detection/data/input/'
sub = pd.read_csv(f'{folder_path}sample_submission.csv')

##### whether use pickle data or not #####
logging.debug('\n\n=== used feature mode =========')
feature_mode = config['feature_mode']
logging.debug(feature_mode)


if feature_mode == 'pickle':
    train = pd.read_pickle('train_2.pkl')
    test = pd.read_pickle('test_2.pkl')
    train = train.drop(['uid', 'uid2', 'uid3', 'uid4', 'uid5', 'bank_type'], axis=1)
    test = test.drop(['uid', 'uid2', 'uid3', 'uid4', 'uid5', 'bank_type'], axis=1)

    drop_cols = ['uid_D1_mean', 'uid2_D1_mean', 'uid2_D1_std', 'uid3_D1_mean', 'uid3_D1_std', 'uid4_D1_mean',
                 'uid4_D1_std', 'uid5_D1_mean', 'uid5_D1_std', 'bank_type_D1_mean', 'bank_type_D1_std', 'uid_D2_mean',
                 'uid_D2_std', 'uid2_D2_mean', 'uid2_D2_std', 'uid3_D2_mean', 'uid3_D2_std', 'uid5_D2_mean',
                 'uid5_D2_std', 'bank_type_D2_mean', 'bank_type_D2_std', 'uid_D3_mean', 'uid2_D3_mean', 'uid2_D3_std',
                 'uid3_D3_mean', 'uid3_D3_std', 'uid4_D3_mean', 'uid4_D3_std', 'uid5_D3_mean', 'uid5_D3_std',
                 'bank_type_D3_mean', 'bank_type_D3_std', 'uid_D4_mean', 'uid_D4_std', 'uid2_D4_mean', 'uid2_D4_std',
                 'uid3_D4_mean', 'uid3_D4_std', 'uid4_D4_mean', 'uid4_D4_std', 'uid5_D4_mean', 'uid5_D4_std',
                 'bank_type_D4_mean', 'bank_type_D4_std', 'uid_D5_mean', 'uid_D5_std', 'uid2_D5_mean', 'uid2_D5_std',
                 'uid3_D5_std', 'uid4_D5_mean', 'uid4_D5_std', 'uid5_D5_mean', 'uid5_D5_std', 'bank_type_D5_mean',
                 'bank_type_D5_std', 'uid_D6_mean', 'uid_D6_std', 'uid2_D6_mean', 'uid2_D6_std', 'uid3_D6_mean',
                 'uid3_D6_std', 'uid4_D6_mean', 'uid4_D6_std', 'uid5_D6_mean', 'uid5_D6_std', 'bank_type_D6_mean',
                 'bank_type_D6_std', 'uid_D7_mean', 'uid_D7_std', 'uid2_D7_mean', 'uid2_D7_std', 'uid3_D7_mean',
                 'uid3_D7_std', 'uid4_D7_mean', 'uid4_D7_std', 'uid5_D7_mean', 'uid5_D7_std', 'bank_type_D7_mean',
                 'uid_D8_std', 'uid2_D8_std', 'uid3_D8_mean', 'uid4_D8_std', 'uid5_D8_std', 'bank_type_D8_mean',
                 'bank_type_D8_std', 'uid_D9_mean', 'uid_D9_std', 'uid2_D9_mean', 'uid2_D9_std', 'uid3_D9_mean',
                 'uid3_D9_std', 'uid4_D9_mean', 'uid4_D9_std', 'uid5_D9_mean', 'uid5_D9_std', 'bank_type_D9_mean',
                 'bank_type_D9_std', 'uid_D10_mean', 'uid2_D10_mean', 'uid2_D10_std', 'uid3_D10_mean', 'uid3_D10_std',
                 'uid4_D10_mean', 'uid5_D10_mean', 'uid5_D10_std', 'bank_type_D10_mean', 'bank_type_D10_std',
                 'uid_D11_mean', 'uid2_D11_mean', 'uid2_D11_std', 'uid3_D11_mean', 'uid3_D11_std', 'uid4_D11_mean',
                 'uid4_D11_std', 'uid5_D11_std', 'bank_type_D11_mean', 'bank_type_D11_std', 'uid_D12_mean',
                 'uid_D12_std', 'uid2_D12_mean', 'uid2_D12_std', 'uid3_D12_mean', 'uid3_D12_std', 'uid4_D12_mean',
                 'uid4_D12_std', 'uid5_D12_mean', 'uid5_D12_std', 'bank_type_D12_std', 'uid_D13_mean', 'uid_D13_std',
                 'uid2_D13_mean', 'uid2_D13_std', 'uid3_D13_mean', 'uid3_D13_std', 'uid4_D13_mean', 'uid4_D13_std',
                 'uid5_D13_mean', 'uid5_D13_std', 'bank_type_D13_std', 'uid_D14_mean', 'uid2_D14_std', 'uid3_D14_std',
                 'uid4_D14_mean', 'uid4_D14_std', 'uid5_D14_mean', 'bank_type_D14_mean', 'bank_type_D14_std',
                 'uid_D15_mean', 'uid_D15_std', 'uid2_D15_mean', 'uid2_D15_std', 'uid3_D15_mean', 'uid3_D15_std',
                 'uid4_D15_mean', 'uid4_D15_std', 'uid5_D15_mean', 'bank_type_D15_mean', 'bank_type_D15_std',
                 'D1_scaled', 'D2_scaled', 'D9_not_na', 'D8_not_same_day', 'D8_D9_decimal_dist', 'D1_DT_D_min_max',
                 'D1_DT_D_std_score', 'D2_DT_D_min_max', 'D2_DT_D_std_score', 'D3_DT_D_min_max', 'D3_DT_D_std_score',
                 'D4_DT_D_min_max', 'D4_DT_D_std_score', 'D5_DT_D_min_max', 'D5_DT_D_std_score', 'D6_DT_D_min_max',
                 'D6_DT_D_std_score', 'D7_DT_D_min_max', 'D7_DT_D_std_score', 'D8_DT_D_min_max', 'D8_DT_D_std_score',
                 'D9_DT_D_min_max', 'D9_DT_D_std_score', 'D10_DT_D_min_max', 'D10_DT_D_std_score', 'D11_DT_D_min_max',
                 'D11_DT_D_std_score', 'D12_DT_D_min_max', 'D12_DT_D_std_score', 'D13_DT_D_min_max',
                 'D13_DT_D_std_score', 'D14_DT_D_min_max', 'D14_DT_D_std_score', 'D15_DT_D_min_max', 'D1_DT_W_min_max',
                 'D1_DT_W_std_score', 'D2_DT_W_std_score', 'D3_DT_W_min_max', 'D3_DT_W_std_score', 'D4_DT_W_min_max',
                 'D4_DT_W_std_score', 'D5_DT_W_min_max', 'D5_DT_W_std_score', 'D6_DT_W_min_max', 'D6_DT_W_std_score',
                 'D7_DT_W_min_max', 'D7_DT_W_std_score', 'D8_DT_W_min_max', 'D8_DT_W_std_score', 'D9_DT_W_min_max',
                 'D9_DT_W_std_score', 'D10_DT_W_min_max', 'D10_DT_W_std_score', 'D11_DT_W_min_max',
                 'D11_DT_W_std_score', 'D12_DT_W_std_score', 'D13_DT_W_min_max', 'D13_DT_W_std_score',
                 'D14_DT_W_min_max', 'D14_DT_W_std_score', 'D15_DT_W_min_max', 'D15_DT_W_std_score', 'D1_DT_M_min_max',
                 'D1_DT_M_std_score', 'D2_DT_M_std_score', 'D3_DT_M_min_max', 'D3_DT_M_std_score', 'D4_DT_M_min_max',
                 'D4_DT_M_std_score', 'D5_DT_M_min_max', 'D5_DT_M_std_score', 'D6_DT_M_min_max', 'D6_DT_M_std_score',
                 'D7_DT_M_min_max', 'D7_DT_M_std_score', 'D8_DT_M_min_max',  'D8_DT_M_std_score', 'D9_DT_M_min_max']

    for col in drop_cols:
        if col in list(train):
            train = train.drop(col, axis=1)
            test = test.drop(col, axis=1)


    # add feature
    logging.debug('\n\n=== stacking =========')
    stacking = config['stacking']
    logging.debug(stacking)
    if stacking == 'true':
        train_add = pd.read_csv('nn_oof_train.csv')
        test_add = pd.read_csv('nn_oof_test.csv')
        train['oof'] = train_add['oof']
        test['oof'] = test_add['isFraud']


else:
    logging.debug('\n\n=== features =========')
    feats = config['features']
    logging.debug(feats)

    # featherからデータの読み込み
    train, test = load_datasets(feats)
    # Noneをnp.nanに戻す
    train.replace(to_replace=[None], value=np.nan, inplace=True)
    test.replace(to_replace=[None], value=np.nan, inplace=True)


    train[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = train['P_emaildomain'].str.split('.', expand=True)
    train[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = train['R_emaildomain'].str.split('.', expand=True)
    test[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = test['P_emaildomain'].str.split('.', expand=True)
    test[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = test['R_emaildomain'].str.split('.', expand=True)

    # # card1のノイズを除去
    # valid_card = train['card1'].value_counts()
    # valid_card = valid_card[valid_card>10]
    # valid_card = list(valid_card.index)
    # train['card1'] = np.where(train['card1'].isin(valid_card), train['card1'], np.nan)
    # test['card1']  = np.where(test['card1'].isin(valid_card), test['card1'], np.nan)


    # dropするカラムの指定
    # one_value_cols = [col for col in train.columns if train[col].nunique() <= 1]
    # one_value_cols_test = [col for col in test.columns if test[col].nunique() <= 1]
    # cols_to_drop = list(set(one_value_cols + one_value_cols_test))
    cols_to_drop=[]
    high_cor_coef = ['C11',
                     'C12',
                     'C14',
                     'D2',
                     'V11',
                     'V140',
                     'V150',
                     'V156',
                     'V157',
                     'V158',
                     'V159',
                     'V160',
                     'V163',
                     'V182',
                     'V189',
                     'V190',
                     'V193',
                     'V196',
                     'V197',
                     'V198',
                     'V199',
                     'V201',
                     'V21',
                     'V212',
                     'V213',
                     'V22',
                     'V222',
                     'V225',
                     'V232',
                     'V233',
                     'V237',
                     'V244',
                     'V247',
                     'V251',
                     'V252',
                     'V254',
                     'V256',
                     'V259',
                     'V26',
                     'V269',
                     'V272',
                     'V273',
                     'V28',
                     'V292',
                     'V298',
                     'V30',
                     'V304',
                     'V317',
                     'V318',
                     'V32',
                     'V329',
                     'V33',
                     'V330',
                     'V331',
                     'V332',
                     'V333',
                     'V43',
                     'V46',
                     'V47',
                     'V49',
                     'V5',
                     'V52',
                     'V58',
                     'V60',
                     'V70',
                     'V72',
                     'V74',
                     'V81',
                     'V9',
                     'V91',
                     'V93',
                     'V94',
                     'id_17',
                     'V107']
    cols_to_drop += high_cor_coef
    # cols_to_drop += ['DT', 'DT_M', 'DT_W', 'DT_D', 'DT_hour', 'DT_day_week', 'DT_day_month', 'DT_M_total',
    #                  'DT_W_total', 'DT_D_total', 'uid', 'uid2', 'uid3', 'uid4', 'uid5', 'bank_type']

    # cols_to_drop += ["TransactionAmt_to_mean_card2", 'TransactionAmt_to_mean_card3', 'TransactionAmt_to_mean_card5',
    #                  'TransactionAmt_to_mean_card6']
    # cols_to_drop += ['V241', 'id_22', 'C3_valid', 'V117', 'V118', 'V120', 'V121', 'V122', 'V305', 'V119']

    # cols_to_drop.remove('isFraud')
    cols_to_drop += ['V' + str(i) for i in range(1, 340)]
    # cols_to_drop += ['uid', 'uid2', 'uid3', 'uid4', 'uid5', 'bank_type']
    cols_to_drop = list(set(cols_to_drop))
    cols_to_drop = list(set(cols_to_drop) & set(list(train)))
    train = train.drop(cols_to_drop, axis=1)
    test = test.drop(cols_to_drop, axis=1)

    print('\nTo drop column : ', cols_to_drop)
    logging.debug('\n\n=== Dropped columns =========')
    logging.debug(cols_to_drop)


    # # 使うかどうか考え中
    # train['D8'] = train['D8'].fillna(-1).astype(int)
    # test['D8'] = test['D8'].fillna(-1).astype(int)
    # train['TransactionAmt'] = train['TransactionAmt'].clip(0, 5000)
    # test['TransactionAmt'] = test['TransactionAmt'].clip(0, 5000)
    # train_df['TransactionAmt'] = np.log1p(train_df['TransactionAmt'])
    # test_df['TransactionAmt'] = np.log1p(test_df['TransactionAmt'])


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
                'M9',
                'P_emaildomain_1',
                'P_emaildomain_2',
                'P_emaildomain_3',
                'R_emaildomain_1',
                'R_emaildomain_2',
                'R_emaildomain_3',
                'OS_id_30',
                'browser_id_31',
                'device_name',
                'device_version',
                'version_id_30',
                'version_id_31',
                #             'Transaction_day_of_week',
                #             'Transaction_hour',
                'P_emaildomain_bin',
                'P_emaildomain_suffix',
                'R_emaildomain_bin',
                'R_emaildomain_suffix',
                "M_sum",
                "M_na",
                "TransactionAmt_3rd_decimal_bin",
                "uid",
                "uid1",
                "uid2",
                "uid3",
                "uid4",
                "uid5",
                "bank_type"
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

            # train[col] = train[col].astype('category')
            # test[col] = test[col].astype('category')

    cat_cols = config['cat_cols']
    # cat_cols = list(set(cat_cols) - set(cols_to_drop))
    logging.debug('\n\n=== categorycal features =========')
    logging.debug(cat_cols)

    for col in list(train):
        if train[col].dtype.name != 'category':
            cur_dominator = list(train[col].fillna(-999).value_counts())[0]
            if (cur_dominator / total_items > 0.85) and (col not in keep_cols):
                cur_dominator = train[col].fillna(-999).value_counts().index[0]
                print('Column:', col, ' | Dominator:', cur_dominator)
                train[col] = np.where(train[col].fillna(-999) == cur_dominator, 1, 0)
                test[col] = np.where(test[col].fillna(-999) == cur_dominator, 1, 0)

                train[col] = train[col].fillna(-999).astype(int)
                test[col] = test[col].fillna(-999).astype(int)

                if col not in cat_cols:
                    cat_cols.append(col)

    print(cat_cols)


    cols_to_drop = list(set(cols_to_drop) & set(list(train)))
    train = train.drop(cols_to_drop, axis=1)
    test = test.drop(cols_to_drop, axis=1)

    ########################### Remove 100% duplicated columns
    cols_sum = {}
    bad_types = ['datetime64[ns]', 'category', 'object']
    for col in list(train):
        if train[col].dtype.name not in bad_types:
            cur_col = train[col].values
            cur_sum = cur_col.mean()
            try:
                cols_sum[cur_sum].append(col)
            except:
                cols_sum[cur_sum] = [col]

    cols_sum = {k: v for k, v in cols_sum.items() if len(v) > 1}

    for k, v in cols_sum.items():
        for col in v[1:]:
            if train[v[0]].equals(train[col]):
                print('Duplicate', col)
                del train[col], test[col]
                if col in cat_cols:
                    cat_cols.remove(col)

    print(cat_cols)

    # add feature
    logging.debug('\n\n=== add features =========')
    add_feats = config['add_feature']
    logging.debug(add_feats)
    train_add, test_add = load_datasets(add_feats)
    train = pd.concat([train, train_add], axis=1)
    test = pd.concat([test, test_add], axis=1)

    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)
    train.to_pickle('train_2.pkl')
    test.to_pickle('test_2.pkl')

    sys.exit()


# Training params
model_type = config['model_type']
logging.debug('\n\n=== model type =========')
logging.debug(model_type)

cat_cols = config['cat_cols']
# cat_cols = list(set(cat_cols) - set(cols_to_drop))
logging.debug('\n\n=== categorycal features =========')
logging.debug(cat_cols)

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


if model_type == 'cat':
    for col in train:
        if train[col].dtype.name != 'category' and col != 'isFraud':
            train[col] = train[col].fillna(-999)
            test[col] = test[col].fillna(-999)

    for col in train:
        if train[col].dtype.name == "category":
            train[col] = train[col].astype("object")
            test[col] = test[col].astype("object")

# Bagging List
y_preds = []
score_list = []
oof_list = []

X_test = test.sort_values('TransactionDT').drop(['TransactionDT', 'TransactionID'], axis=1)
test = test[["TransactionDT", 'TransactionID']]

for i in range(bag_times):
    if dwsp_rate != 1:

        train_fr = train.loc[train['isFraud'] == 1]
        train_nofr = train.loc[train['isFraud'] == 0]
        del train
        train_nofr_dwsp = train_nofr.sample(n=math.floor(train_nofr.shape[0] * dwsp_rate), replace=True, random_state=i)
        train_dwsp = pd.concat([train_fr, train_nofr_dwsp]).sample(frac=1, random_state=i)
        X = train_dwsp.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
        y = train_dwsp.sort_values('TransactionDT')['isFraud']
        del train_dwsp
        gc.collect()
        print('\n Down sampling is completed.\n')
        print(
            '---------- iter {} start -------------------------------------'
            '------------------------------------------------------------'.format(str(i + 1)))
        result_dict_lgb = train_model_classification(
            X=X,
            X_test=X_test,
            y=y,
            params=params,
            folds=folds,
            model_type=model_type,
            eval_metric='auc',
            plot_feature_importance=True,
            verbose=500,
            early_stopping_rounds=100,
            n_estimators=10000,
            averaging='usual',
            n_jobs=-1,
            cat_cols=cat_cols,
            seed=i,
            groups=split_groups
        )

        y_preds.append(result_dict_lgb['prediction'])
        score_list.append(np.mean(result_dict_lgb['scores']))
        gc.collect()

    else:
        X = train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
        y = train.sort_values('TransactionDT')['isFraud']
        train = train.sort_values('TransactionDT')['TransactionID']
        gc.collect()
        print('\n Use full data.\n')

        for k in range(random_seed_average_times):
            print(
                '---------- iter {} start -------------------------------------'
                '------------------------------------------------------------'.format(str(k + 1)))
            result_dict_lgb = train_model_classification(
                X=X,
                X_test=X_test,
                y=y,
                params=params,
                folds=folds,
                model_type=model_type,
                eval_metric='auc',
                plot_feature_importance=True,
                verbose=500,
                early_stopping_rounds=100,
                n_estimators=10000,
                averaging='usual',
                n_jobs=-1,
                cat_cols=cat_cols,
                seed=k,
                groups=split_groups
            )

            y_preds.append(result_dict_lgb['prediction'])
            score_list.append(np.mean(result_dict_lgb['scores']))
            oof_list.append(result_dict_lgb['oof'])
            gc.collect()

    # 特徴量の選択
    # df_best_features = pd.DataFrame()
    # df_best_features = pd.concat([df_best_features, result_dict_lgb['feature_importance']], axis=0)
    # df_best_features.sort_values(by="importance", ascending=False).head()
    # good_columns = list(df_best_features[["feature", "importance"]].groupby("feature").mean().sort_values(
    #     by="importance", ascending=False).index)
    #
    # cut_off = math.floor(X.shape[1] * 0.9)
    # cut_off_column = good_columns[cut_off:]
    # print(cut_off_column)
    # logging.debug('\n\n=== 10% Worst features =========')
    # logging.debug(cut_off_column)


print(
    '--------------------------------------------------------------'
    '------------------------------------------------------------------')

y_preds_bagging = sum(y_preds) / len(y_preds)

test['prediction'] = y_preds_bagging
score = np.mean(score_list)

logging.debug('\n\n=== CV scores =========')
logging.debug(score)

# submission
sub['isFraud'] = pd.merge(sub, test, on='TransactionID')['prediction']
sub.to_csv(
    '/tmp/working/IEEE_Fraud_Detection/data/output/sub_{0:%Y%m%d%H%M}_{1}_{2:.4f}.csv'.format(
        now,
        model_type,
        score),
    index=False)

if len(oof_list) > 0:
    oof_bagging = sum(oof_list) / len(oof_list)
    train['oof'] = oof_bagging
    train[['TransactionID', 'oof']].to_pickle(
        '/tmp/working/IEEE_Fraud_Detection/data/output/train_oof_{0:%Y%m%d%H%M}_{1}_{2:.4f}.pkl'.format(
            now,
            model_type,
            score))
    test['oof'] = test['prediction']
    test[['TransactionID', 'oof']].to_pickle(
        '/tmp/working/IEEE_Fraud_Detection/data/output/test_oof_{0:%Y%m%d%H%M}_{1}_{2:.4f}.pkl'.format(
            now,
            model_type,
            score))




print("\n'run.py' completed.")

# calculate exectime
t2 = time.time()
exec_time = math.floor((t2 - t1) / 60)
print('execution time: {} min'.format(str(exec_time)), '\n')
logging.debug('\n\n=== Execution Time =========')
logging.debug('{} min'.format(str(exec_time)))
