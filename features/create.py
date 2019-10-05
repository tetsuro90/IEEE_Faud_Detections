import pandas as pd
import numpy as np
import re as re
import sys, gc, datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

sys.path.append('/tmp/working/IEEE_Fraud_Detection/')
from features.base import Feature, get_arguments, generate_features
from functions.functions import load_datasets, reduce_mem_usage, uid_aggregation, values_normalization, timeblock_frequency_encoding

Feature.dir = '/tmp/working/IEEE_Fraud_Detection/features'

######################################################################################################
################# create features ####################################################################
######################################################################################################
# base DataFrame


class TransactionIdentityMerged(Feature):
    def create_features(self):
        self.train[list(train.columns.values)] = train
        self.test[list(test.columns.values)] = test

#####################################################################################
#### reference: https://www.kaggle.com/artgor/eda-and-models#Feature-engineering ####


class TransactionAmtToMeanCard1(Feature):
    def create_features(self):
        self.train['TransactionAmt_to_mean_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('mean')
        self.test['TransactionAmt_to_mean_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('mean')


class TransactionAmtToMeanCard4(Feature):
    def create_features(self):
        self.train['TransactionAmt_to_mean_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('mean')
        self.test['TransactionAmt_to_mean_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('mean')


class TransactionAmtToStdCard1(Feature):
    def create_features(self):
        self.train['TransactionAmt_to_std_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('std')
        self.test['TransactionAmt_to_std_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('std')


class TransactionAmtToStdCard4(Feature):
    def create_features(self):
        self.train['TransactionAmt_to_std_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('std')
        self.test['TransactionAmt_to_std_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('std')


class TransactionAmtToMeanStdAllcard(Feature):
    def create_features(self):
        cols = ['card2', 'card3', 'card5', 'card6']
        for col in cols:
            for agg in ['mean', 'std']:
                self.train['TransactionAmt_to_' + agg + '_' + col] = \
                    train['TransactionAmt'] / train.groupby([col])['TransactionAmt'].transform(agg)
                self.test['TransactionAmt_to_' + agg + '_' + col] = \
                    test['TransactionAmt'] / test.groupby([col])['TransactionAmt'].transform(agg)


class TransactionAmtToMeanStdAddr(Feature):
    def create_features(self):
        cols = ['addr1', 'addr2']
        for col in cols:
            for agg in ['mean', 'std']:
                self.train['TransactionAmt_to_' + agg + '_' + col] = \
                    train['TransactionAmt'] / train.groupby([col])['TransactionAmt'].transform(agg)
                self.test['TransactionAmt_to_' + agg + '_' + col] = \
                    test['TransactionAmt'] / test.groupby([col])['TransactionAmt'].transform(agg)


class Id02ToMeanCard1(Feature):
    def create_features(self):
        self.train['id_02_to_mean_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('mean')
        self.test['id_02_to_mean_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('mean')


class Id02ToMeanCard4(Feature):
    def create_features(self):
        self.train['id_02_to_mean_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('mean')
        self.test['id_02_to_mean_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('mean')


class Id02ToStdCard1(Feature):
    def create_features(self):
        self.train['id_02_to_std_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('std')
        self.test['id_02_to_std_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('std')


class Id02ToStdCard4(Feature):
    def create_features(self):
        self.train['id_02_to_std_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('std')
        self.test['id_02_to_std_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('std')


class D15ToMeanCard1(Feature):
    def create_features(self):
        self.train['D15_to_mean_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('mean')
        self.test['D15_to_mean_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('mean')


class D15ToMeanCard4(Feature):
    def create_features(self):
        self.train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')
        self.test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')


class D15ToStdCard1(Feature):
    def create_features(self):
        self.train['D15_to_std_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('std')
        self.test['D15_to_std_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('std')


class D15ToStdCard4(Feature):
    def create_features(self):
        self.train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')
        self.test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')


class D15ToMeanAddr1(Feature):
    def create_features(self):
        self.train['D15_to_mean_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('mean')
        self.test['D15_to_mean_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('mean')


class D15ToStdAddr1(Feature):
    def create_features(self):
        self.train['D15_to_std_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('std')
        self.test['D15_to_std_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('std')


class D15ToMeanAddr2(Feature):
    def create_features(self):
        self.train['D15_to_mean_addr2'] = train['D15'] / train.groupby(['addr2'])['D15'].transform('mean')
        self.test['D15_to_mean_addr2'] = test['D15'] / test.groupby(['addr2'])['D15'].transform('mean')


class D15ToStdAddr2(Feature):
    def create_features(self):
        self.train['D15_to_std_addr2'] = train['D15'] / train.groupby(['addr2'])['D15'].transform('std')
        self.test['D15_to_std_addr2'] = test['D15'] / test.groupby(['addr2'])['D15'].transform('std')


class EmaildomainP(Feature):
    def create_features(self):
        train[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = train['P_emaildomain'].str.split('.', expand=True)
        test[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = test['P_emaildomain'].str.split('.', expand=True)
        for col in ['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']:
            if train[col].dtype == object:
                train[col].replace(to_replace=np.nan, value='-999', inplace=True)
            else:
                train[col].replace(to_replace=np.nan, value= -999, inplace=True)
        for col in ['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']:
            if test[col].dtype == object:
                test[col].replace(to_replace=np.nan, value='-999', inplace=True)
            else:
                test[col].replace(to_replace=np.nan, value= -999, inplace=True)

        self.train[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = train[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']]
        self.test[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = test[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']]


class EmaildomainR(Feature):
    def create_features(self):
        train[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = train['R_emaildomain'].str.split('.', expand=True)
        test[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = test['R_emaildomain'].str.split('.', expand=True)
        for col in ['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']:
            if train[col].dtype == object:
                train[col].replace(to_replace=np.nan, value='-999', inplace=True)
            else:
                train[col].replace(to_replace=np.nan, value= -999, inplace=True)
        for col in ['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']:
            if test[col].dtype == object:
                test[col].replace(to_replace=np.nan, value='-999', inplace=True)
            else:
                test[col].replace(to_replace=np.nan, value= -999, inplace=True)
        self.train[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = train[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']]
        self.test[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = test[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']]


class EmaildomainProton(Feature):
    def create_features(self):
        self.train['P_isproton'] = np.where(train['P_emaildomain'] == 'protonmail.com', 1, 0)
        self.train['R_isproton'] = np.where(train['R_emaildomain'] == 'protonmail.com', 1, 0)
        self.test['P_isproton'] = np.where(test['P_emaildomain'] == 'protonmail.com', 1, 0)
        self.test['R_isproton'] = np.where(test['R_emaildomain'] == 'protonmail.com', 1, 0)



#####################################################################################
# reference: https://www.kaggle.com/davidcairuz/feature-engineering-lightgbm-corrected
#####################################################################################


class DeviceName(Feature):
    def create_features(self):
        def id_split(dataframe):
            dataframe['device_name'] = dataframe['DeviceInfo'].str.split('/', expand=True)[0]
            dataframe.loc[dataframe['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
            dataframe.loc[dataframe['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
            dataframe.loc[dataframe['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
            dataframe.loc[dataframe['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
            dataframe.loc[dataframe['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
            dataframe.loc[dataframe['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
            dataframe.loc[dataframe['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
            dataframe.loc[dataframe['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
            dataframe.loc[dataframe['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
            dataframe.loc[dataframe['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
            dataframe.loc[dataframe['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
            dataframe.loc[dataframe['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
            dataframe.loc[dataframe['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
            dataframe.loc[dataframe['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
            dataframe.loc[dataframe['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
            dataframe.loc[dataframe['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
            dataframe.loc[dataframe['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'

            dataframe.loc[dataframe.device_name.isin(dataframe.device_name.value_counts()[dataframe.device_name.value_counts() < 200].index), 'device_name'] = "Others"
            gc.collect()

            return dataframe['device_name']

        self.train['device_name'] = id_split(train)
        self.test['device_name'] = id_split(test)



class DeviceVersion(Feature):
    def create_features(self):
        self.train['device_version'] = train['DeviceInfo'].str.split('/', expand=True)[1]
        self.test['device_version'] = test['DeviceInfo'].str.split('/', expand=True)[1]



class OsId30(Feature):
    def create_features(self):
        self.train['OS_id_30'] = train['id_30'].str.split(' ', expand=True)[0]
        self.test['OS_id_30'] = test['id_30'].str.split(' ', expand=True)[0]



class VersionId30(Feature):
    def create_features(self):
        self.train['version_id_30'] = train['id_30'].str.split(' ', expand=True)[1]
        self.test['version_id_30'] = test['id_30'].str.split(' ', expand=True)[1]



class BrowserId31(Feature):
    def create_features(self):
        self.train['browser_id_31'] = train['id_31'].str.split(' ', expand=True)[0]
        self.test['browser_id_31'] = test['id_31'].str.split(' ', expand=True)[0]



class VersionId31(Feature):
    def create_features(self):
        self.train['version_id_31'] = train['id_31'].str.split(' ', expand=True)[1]
        self.test['version_id_31'] = test['id_31'].str.split(' ', expand=True)[1]



class ScreenWidth(Feature):
    def create_features(self):
        self.train['screen_width'] = train['id_33'].str.split('x', expand=True)[0].astype(float)
        self.test['screen_width'] = test['id_33'].str.split('x', expand=True)[0].astype(float)



class ScreenHeight(Feature):
    def create_features(self):
        self.train['screen_height'] = train['id_33'].str.split('x', expand=True)[1].astype(float)
        self.test['screen_height'] = test['id_33'].str.split('x', expand=True)[1].astype(float)



class TransactionAmtLog(Feature):
    def create_features(self):
        self.train['TransactionAmt_Log'] = np.log(train['TransactionAmt'])
        self.test['TransactionAmt_Log'] = np.log(test['TransactionAmt'])



class TrasactopmAmtDecimal(Feature):
    def create_features(self):
        self.train['TransactionAmt_decimal'] = \
            ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(int)
        self.test['TransactionAmt_decimal'] = \
            ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)


class TrasactopmAmtDecimalThird(Feature):
    def create_features(self):
        train['TransactionAmt_3rd_decimal'] = train['TransactionAmt'].astype(str).str.split('.', expand=True)[1].str[2]
        test['TransactionAmt_3rd_decimal'] = test['TransactionAmt'].astype(str).str.split('.', expand=True)[1].str[2]
        train['TransactionAmt_3rd_decimal'].replace(np.nan, 'missing', inplace=True)
        test['TransactionAmt_3rd_decimal'].replace(np.nan, 'missing', inplace=True)
        self.train['TransactionAmt_3rd_decimal_bin'] = np.where(train['TransactionAmt_3rd_decimal'] == 'missing', 0, 1)
        self.test['TransactionAmt_3rd_decimal_bin'] = np.where(train['TransactionAmt_3rd_decimal'] == 'missing', 0, 1)


class TrasactionDayOfWeek(Feature):
    def create_features(self):
        self.train['Transaction_day_of_week'] = np.floor((train['TransactionDT'] / (3600 * 24) - 1) % 7)
        self.test['Transaction_day_of_week'] = np.floor((test['TransactionDT'] / (3600 * 24) - 1) % 7)






class TrasactionHour(Feature):
    def create_features(self):
        self.train['Transaction_hour'] = np.floor(train['TransactionDT'] / 3600) % 24
        self.test['Transaction_hour'] = np.floor(test['TransactionDT'] / 3600) % 24



class FeaturesInteraction(Feature):
    def create_features(self):
        for feature in ['id_02__id_20', 'id_02__D8', 'D11__DeviceInfo', 'DeviceInfo__P_emaildomain',
                        'P_emaildomain__C2', 'card2__dist1', 'card1__card5', 'card2__id_20',
                        'card5__P_emaildomain', 'addr1__card1']:
            f1, f2 = feature.split('__')
            train[feature] = train[f1].astype(str) + '_' + train[f2].astype(str)
            test[feature] = test[f1].astype(str) + '_' + test[f2].astype(str)
            le = LabelEncoder()
            le.fit(list(train[feature].astype(str).values) + list(test[feature].astype(str).values))
            self.train[feature] = le.transform(list(train[feature].astype(str).values))
            self.test[feature] = le.transform(list(test[feature].astype(str).values))


###########################################
######## CountEncoding ####################
###########################################

class CountEncodingBoth(Feature):
    def create_features(self):
        for feature in ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'id_36']:
            self.train[feature + '_count_full'] = \
                train[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))
            self.test[feature + '_count_full'] = \
                test[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))


class CountEncodingBoth2(Feature):
    def create_features(self):
        for feature in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
                        'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9',
                        'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain']:
            self.train[feature + '_count_full'] = \
                train[feature].map(pd.concat([train[feature], test[feature]],
                                             ignore_index=True).value_counts(dropna=False))
            self.test[feature + '_count_full'] = \
                test[feature].map(pd.concat([train[feature], test[feature]],
                                            ignore_index=True).value_counts(dropna=False))


class CountEncodingBoth3(Feature):
    def create_features(self):
        def id_split(dataframe):
            dataframe['device_name'] = dataframe['DeviceInfo'].str.split('/', expand=True)[0]
            dataframe.loc[dataframe['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
            dataframe.loc[dataframe['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
            dataframe.loc[dataframe['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
            dataframe.loc[dataframe['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
            dataframe.loc[dataframe['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
            dataframe.loc[dataframe['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
            dataframe.loc[dataframe['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
            dataframe.loc[dataframe['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
            dataframe.loc[dataframe['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
            dataframe.loc[dataframe['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
            dataframe.loc[dataframe['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
            dataframe.loc[dataframe['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
            dataframe.loc[dataframe['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
            dataframe.loc[dataframe['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
            dataframe.loc[dataframe['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
            dataframe.loc[dataframe['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
            dataframe.loc[dataframe['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'

            dataframe.loc[dataframe.device_name.isin(dataframe.device_name.value_counts()[dataframe.device_name.value_counts() < 200].index), 'device_name'] = "Others"
            gc.collect()

            return dataframe['device_name']

        train['device_name'] = id_split(train)
        test['device_name'] = id_split(test)

        train['device_version'] = train['DeviceInfo'].str.split('/', expand=True)[1]
        test['device_version'] = test['DeviceInfo'].str.split('/', expand=True)[1]

        train['OS_id_30'] = train['id_30'].str.split(' ', expand=True)[0]
        test['OS_id_30'] = test['id_30'].str.split(' ', expand=True)[0]

        train['version_id_30'] = train['id_30'].str.split(' ', expand=True)[1]
        test['version_id_30'] = test['id_30'].str.split(' ', expand=True)[1]

        train['browser_id_31'] = train['id_31'].str.split(' ', expand=True)[0]
        test['browser_id_31'] = test['id_31'].str.split(' ', expand=True)[0]

        train['version_id_31'] = train['id_31'].str.split(' ', expand=True)[1]
        test['version_id_31'] = test['id_31'].str.split(' ', expand=True)[1]

        for feature in ['DeviceInfo', 'device_name', 'device_version',
                        'OS_id_30', 'version_id_30', 'browser_id_31', 'version_id_31']:
            self.train[feature + '_count_full'] = \
                train[feature].map(pd.concat([train[feature], test[feature]],
                                             ignore_index=True).value_counts(dropna=False))
            self.test[feature + '_count_full'] = \
                test[feature].map(pd.concat([train[feature], test[feature]],
                                            ignore_index=True).value_counts(dropna=False))


class CountEncodingBoth4(Feature):
    def create_features(self):
        for feature in ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08', 'id_09', 'id_10',
                        'id_11', 'id_13', 'id_14', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_24',
                        'id_25', 'id_26', 'id_30', 'id_31', 'id_32', 'id_33', 'id_36']:
            self.train[feature + '_count_full'] = \
                train[feature].map(pd.concat([train[feature], test[feature]],
                                             ignore_index=True).value_counts(dropna=False))
            self.test[feature + '_count_full'] = \
                test[feature].map(pd.concat([train[feature], test[feature]],
                                            ignore_index=True).value_counts(dropna=False))


class CountEncodingBoth5(Feature):
    def create_features(self):
        train['uid'] = train['card1'].astype(str) + '_' + train['card2'].astype(str)
        test['uid'] = test['card1'].astype(str) + '_' + test['card2'].astype(str)
        train['uid2'] = train['uid'].astype(str) + '_' + train['card3'].astype(str) + '_' + train['card5'].astype(str)
        test['uid2'] = test['uid'].astype(str) + '_' + test['card3'].astype(str) + '_' + test['card5'].astype(str)
        train['uid3'] = train['uid2'].astype(str) + '_' + train['addr1'].astype(str) + '_' + train['addr2'].astype(str)
        test['uid3'] = test['uid2'].astype(str) + '_' + test['addr1'].astype(str) + '_' + test['addr2'].astype(str)

        for feature in ['uid', 'uid2', 'uid3']:
            self.train[feature + '_count_full'] = \
                train[feature].map(pd.concat([train[feature], test[feature]],
                                             ignore_index=True).value_counts(dropna=False))
            self.test[feature + '_count_full'] = \
                test[feature].map(pd.concat([train[feature], test[feature]],
                                            ignore_index=True).value_counts(dropna=False))


class CountEncodingBothV(Feature):
    def create_features(self):
        for feature in ['V10', 'V11']:
            self.train[feature + '_count_full'] = \
                train[feature].map(pd.concat([train[feature], test[feature]],
                                             ignore_index=True).value_counts(dropna=False))
            self.test[feature + '_count_full'] = \
                test[feature].map(pd.concat([train[feature], test[feature]],
                                            ignore_index=True).value_counts(dropna=False))



class CountEncodingSep(Feature):
    def create_features(self):
        for feature in ['id_01', 'id_31', 'id_33', 'id_36']:
            self.train[feature + '_count_dist'] = train[feature].map(train[feature].value_counts(dropna=False))
            self.test[feature + '_count_dist'] = test[feature].map(test[feature].value_counts(dropna=False))

###########################################


class EmailDomainBinSuffix(Feature):
    def create_features(self):
        emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 'optonline.net': 'other',
                  'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo',
                  'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink',
                  'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other',
                  'hotmail.com': 'microsoft', 'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other', 'outlook.es': 'microsoft',
                  'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other',
                  'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink',
                  'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att',
                  'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other',
                  'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo',
                  'netzero.com': 'other', 'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other', 'aol.com': 'aol',
                  'juno.com': 'other', 'icloud.com': 'apple'}
        us_emails = ['gmail', 'net', 'edu']

        for c in ['P_emaildomain', 'R_emaildomain']:
            self.train[c + '_bin'] = train[c].map(emails)
            self.test[c + '_bin'] = test[c].map(emails)

            train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])
            test[c + '_suffix'] = test[c].map(lambda x: str(x).split('.')[-1])

            self.train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
            self.test[c + '_suffix'] = test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

#####################################################################################
# reference: https://www.kaggle.com/kyakovlev/ieee-gb-2-make-amount-useful-again
#####################################################################################

# Targetencodingはleakに繋がるから気をるけること。


class TargetMean(Feature):
    def create_features(self):
        TARGET = 'isFraud'
        for col in ['ProductCD','M4']:
            temp_train = pd.concat([train[[col]], test[[col]]])
            col_encoded = temp_train[col].value_counts().to_dict()
            train[col] = train[col].map(col_encoded)
            test[col]  = test[col].map(col_encoded)

            temp_dict = train.groupby([col])[TARGET].agg(['mean']).reset_index().rename(
                                                                columns={'mean': col+'_target_mean'})
            temp_dict.index = temp_dict[col].values
            temp_dict = temp_dict[col+'_target_mean'].to_dict()

            self.train[col+'_target_mean'] = train[col].map(temp_dict)
            self.test[col+'_target_mean'] = test[col].map(temp_dict)


class EmailCheck(Feature):
    def create_features(self):
        # train['P_emaildomain'] = train['P_emaildomain'].astype('object')
        # test['P_emaildomain'] = test['P_emaildomain'].astype('object')
        # train['R_emaildomain'] = train['R_emaildomain'].astype('object')
        # test['R_emaildomain'] = test['R_emaildomain'].astype('object')

        new_columns = []

        train['email_check'] = np.where(train['P_emaildomain'] == train['R_emaildomain'], 1, 0)
        test['email_check'] = np.where(test['P_emaildomain'] == test['R_emaildomain'], 1, 0)
        new_columns.append('email_check')

        # All NaNs
        train['email_check_nan_all'] = np.where((train['P_emaildomain'].isna()) &
                                                (train['R_emaildomain'].isna()), 1, 0)
        test['email_check_nan_all'] = np.where((test['P_emaildomain'].isna()) &
                                               (test['R_emaildomain'].isna()), 1, 0)
        new_columns.append('email_check_nan_all')

        # Any NaN
        train['email_check_nan_any'] = np.where((train['P_emaildomain'].isna()) |
                                                (train['R_emaildomain'].isna()), 1, 0)
        test['email_check_nan_any'] = np.where((test['P_emaildomain'].isna()) |
                                               (test['R_emaildomain'].isna()), 1, 0)
        new_columns.append('email_check_nan_any')

        def fix_emails(train):
            train['P_emaildomain'] = train['P_emaildomain'].fillna('email_not_provided')
            train['R_emaildomain'] = train['R_emaildomain'].fillna('email_not_provided')

            train['email_match_not_nan'] = np.where((train['P_emaildomain'] == train['R_emaildomain']) &
                                                    (train['P_emaildomain'] != 'email_not_provided'), 1, 0)
            return train['email_match_not_nan']

        train['email_match_not_nan'] = fix_emails(train)
        test['email_match_not_nan'] = fix_emails(test)
        new_columns.append('email_match_not_nan')

        self.train[new_columns] = train[new_columns]
        self.test[new_columns] = test[new_columns]


class D9AndTransactiondt(Feature):
    def create_features(self):
        train['local_hour'] = train['D9']*24
        test['local_hour'] = test['D9']*24
        self.train['local_hour'] = train['local_hour'] - (train['TransactionDT']/(60*60))%24
        self.test['local_hour'] = test['local_hour'] - (test['TransactionDT']/(60*60))%24
        self.train['local_hour_dist'] = train['local_hour']/train['dist2']
        self.test['local_hour_dist'] = test['local_hour']/test['dist2']


class CategoryMSumNan(Feature):
    def create_features(self):
        for col in ['M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']:
            train[col] = train[col].map({'T': 1, 'F': 0})
            test[col] = test[col].map({'T': 1, 'F': 0})
        i_cols = ['M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']
        self.train['M_sum'] = train[i_cols].sum(axis=1).astype(np.int8)
        self.test['M_sum'] = test[i_cols].sum(axis=1).astype(np.int8)
        self.train['M_na'] = train[i_cols].isna().sum(axis=1).astype(np.int8)
        self.test['M_na'] = test[i_cols].isna().sum(axis=1).astype(np.int8)


class CategoryCSumNanValid(Feature):
    def create_features(self):
        i_cols = ['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14']
        train['C_sum'] = 0
        test['C_sum'] = 0

        train['C_null'] = 0
        test['C_null'] = 0

        for col in i_cols:
            train['C_sum'] += np.where(train[col]==1,1,0)
            test['C_sum'] += np.where(test[col]==1,1,0)
            train['C_null'] += np.where(train[col]==0,1,0)
            test['C_null'] += np.where(test[col]==0,1,0)

            valid_values = train[col].value_counts()
            valid_values = valid_values[valid_values>1000]
            valid_values = list(valid_values.index)

            self.train[col+'_valid'] = np.where(train[col].isin(valid_values),1,0)
            self.test[col+'_valid'] = np.where(test[col].isin(valid_values),1,0)

        self.train['C_sum'] = train['C_sum']
        self.test['C_sum'] = test['C_sum']
        self.train['C_null'] = train['C_null']
        self.test['C_null'] = test['C_null']


class CategoryCardNan(Feature):
    def create_features(self):
        train['uid'] = train['card1'].astype(str) + '_' + train['card2'].astype(str)
        test['uid'] = test['card1'].astype(str) + '_' + test['card2'].astype(str)
        train['uid2'] = train['uid'].astype(str)+'_'+train['card3'].astype(str)+'_'+train['card5'].astype(str)
        test['uid2'] = test['uid'].astype(str)+'_'+test['card3'].astype(str)+'_'+test['card5'].astype(str)
        train['uid3'] = train['uid2'].astype(str)+'_'+train['addr1'].astype(str)+'_'+train['addr2'].astype(str)
        test['uid3'] = test['uid2'].astype(str)+'_'+test['addr1'].astype(str)+'_'+test['addr2'].astype(str)

        i_cols = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']

        self.train['card_nan'] = train[i_cols].isna().sum(axis=1).astype(np.int8)
        self.test['card_nan'] = test[i_cols].isna().sum(axis=1).astype(np.int8)



class BlockVSum1(Feature):
    def create_features(self):
        block = ['V' + str(i + 1) for i in range(11)]
        self.train['block_v_sum1'] = train[block].sum(axis=1).astype(np.int8)
        self.test['block_v_sum1'] = test[block].sum(axis=1).astype(np.int8)



class BlockVNan1(Feature):
    def create_features(self):
        block = ['V' + str(i + 1) for i in range(11)]
        self.train['block_v_nan1'] = train[block].isna().sum(axis=1).astype(np.int8)
        self.test['block_v_nan1'] = test[block].isna().sum(axis=1).astype(np.int8)


###############################################
######## TransactionAmtStd ####################
###############################################

class TransactionAmtMeanStd(Feature):
    def create_features(self):
        train['uid'] = train['card1'].astype(str) + '_' + train['card2'].astype(str)
        test['uid'] = test['card1'].astype(str) + '_' + test['card2'].astype(str)
        train['uid2'] = train['uid'].astype(str)+'_'+train['card3'].astype(str)+'_'+train['card5'].astype(str)
        test['uid2'] = test['uid'].astype(str)+'_'+test['card3'].astype(str)+'_'+test['card5'].astype(str)
        train['uid3'] = train['uid2'].astype(str)+'_'+train['addr1'].astype(str)+'_'+train['addr2'].astype(str)
        test['uid3'] = test['uid2'].astype(str)+'_'+test['addr1'].astype(str)+'_'+test['addr2'].astype(str)


        i_cols = [
            'card1',
            'card2',
            'card3',
            'card5',
            'uid',
            'uid2',
            'uid3'
        ]

        for col in i_cols:
            for agg_type in ['mean', 'std']:
                new_col_name = col+'_TransactionAmt_'+agg_type
                temp_train = pd.concat([train[[col, 'TransactionAmt']], test[[col,'TransactionAmt']]])
                temp_train = temp_train.groupby([col])['TransactionAmt'].agg([agg_type]).reset_index().rename(
                                                        columns={agg_type: new_col_name})

                temp_train.index = list(temp_train[col])
                temp_train = temp_train[new_col_name].to_dict()

                self.train[new_col_name] = train[col].map(temp_train)
                self.test[new_col_name]  = test[col].map(temp_train)


class TransactionAmtMeanStd2(Feature):
    def create_features(self):
        i_cols = [
            'card4',
            'card6',
            'addr1',
            'addr2'
        ]

        for col in i_cols:
            for agg_type in ['mean', 'std']:
                new_col_name = col+'_TransactionAmt_'+agg_type
                temp_train = pd.concat([train[[col, 'TransactionAmt']], test[[col,'TransactionAmt']]])
                temp_train = temp_train.groupby([col])['TransactionAmt'].agg([agg_type]).reset_index().rename(
                                                        columns={agg_type: new_col_name})

                temp_train.index = list(temp_train[col])
                temp_train = temp_train[new_col_name].to_dict()

                self.train[new_col_name] = train[col].map(temp_train)
                self.test[new_col_name]  = test[col].map(temp_train)


class TransactionAmtMeanStdV(Feature):
    def create_features(self):
        i_cols = [
            'V10',
            'V11'
        ]

        for col in i_cols:
            for agg_type in ['mean', 'std']:
                new_col_name = col+'_TransactionAmt_'+agg_type
                temp_train = pd.concat([train[[col, 'TransactionAmt']], test[[col,'TransactionAmt']]])
                temp_train = temp_train.groupby([col])['TransactionAmt'].agg([agg_type]).reset_index().rename(
                                                        columns={agg_type: new_col_name})

                temp_train.index = list(temp_train[col])
                temp_train = temp_train[new_col_name].to_dict()

                self.train[new_col_name] = train[col].map(temp_train)
                self.test[new_col_name]  = test[col].map(temp_train)


class LogTransactionAmtMeanStd(Feature):
    def create_features(self):
        train['uid'] = train['card1'].astype(str) + '_' + train['card2'].astype(str)
        test['uid'] = test['card1'].astype(str) + '_' + test['card2'].astype(str)
        train['uid2'] = train['uid'].astype(str) + '_' + train['card3'].astype(str) + '_' + train['card5'].astype(str)
        test['uid2'] = test['uid'].astype(str) + '_' + test['card3'].astype(str) + '_' + test['card5'].astype(str)
        train['uid3'] = train['uid2'].astype(str) + '_' + train['addr1'].astype(str) + '_' + train['addr2'].astype(str)
        test['uid3'] = test['uid2'].astype(str) + '_' + test['addr1'].astype(str) + '_' + test['addr2'].astype(str)

        i_cols = [
            'card1',
            'card2',
            'card3',
            'card4',
            'card5',
            'card6',  # 効果ありそう
            'uid',  # 効果ありそう
            'uid2',  # 効果ありそう
            'uid3',
            'addr1',
            'addr2'
        ]

        for col in i_cols:
            for agg_type in ['mean', 'std']:
                new_col_name = col+'_TransactionAmt_'+agg_type
                temp_train = pd.concat([train[[col, 'TransactionAmt']], test[[col,'TransactionAmt']]])
                temp_train = temp_train.groupby([col])['TransactionAmt'].agg([agg_type]).reset_index().rename(
                                                        columns={agg_type: new_col_name})

                temp_train.index = list(temp_train[col])
                temp_train = temp_train[new_col_name].to_dict()
                self.train['Log' + new_col_name] = np.log(train[col].map(temp_train))
                self.test['Log' + new_col_name]  = np.log(test[col].map(temp_train))

###########################################


class AddressMatch(Feature):
    def create_features(self):
        train['bank_type'] = train['card3'].astype(str)+'_'+train['card5'].astype(str)
        test['bank_type']  = test['card3'].astype(str)+'_'+test['card5'].astype(str)

        train['address_match'] = train['bank_type'].astype(str)+'_'+train['addr2'].astype(str)
        test['address_match']  = test['bank_type'].astype(str)+'_'+test['addr2'].astype(str)

        for col in ['address_match','bank_type']:
            temp_train = pd.concat([train[[col]], test[[col]]])
            temp_train[col] = np.where(temp_train[col].str.contains('nan'), np.nan, temp_train[col])
            temp_train = temp_train.dropna()
            fq_encode = temp_train[col].value_counts().to_dict()
            train[col] = train[col].map(fq_encode)
            test[col]  = test[col].map(fq_encode)

        self.train['address_match'] = train['address_match']/train['bank_type']
        self.test['address_match']  = test['address_match']/test['bank_type']


###########################################
# reference: https://www.kaggle.com/fchmiel/day-and-time-powerful-predictive-feature
class MakeHourFeature(Feature):
    def create_features(self):
        self.train['hours'] = np.floor(train['TransactionDT'] / 3600) % 24
        self.test['hours'] = np.floor(test['TransactionDT'] / 3600) % 24


###########################################
# referene: https://www.kaggle.com/kyakovlev/ieee-fe-with-some-eda
class DateOfMonth(Feature):
    def create_features(self):
        START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
        train['DT'] = train['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
        self.train['DT_M'] = ((train['DT'].dt.year - 2017) * 12 + train['DT'].dt.month).astype(np.int8)
        test['DT'] = test['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
        self.test['DT_M'] = ((test['DT'].dt.year - 2017) * 12 + test['DT'].dt.month).astype(np.int8)


class TransactionDT(Feature):
    def create_features(self):
        START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
        from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
        dates_range = pd.date_range(start='2017-10-01', end='2019-01-01')
        us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())

        # Let's add temporary "time variables" for aggregations
        # and add normal "time variables"
        self.train['DT'] = train['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
        train['DT'] = train['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
        self.train['DT_M'] = ((train['DT'].dt.year - 2017) * 12 + train['DT'].dt.month).astype(np.int8)
        train['DT_M'] = ((train['DT'].dt.year - 2017) * 12 + train['DT'].dt.month).astype(np.int8)
        self.train['DT_W'] = ((train['DT'].dt.year - 2017) * 52 + train['DT'].dt.weekofyear).astype(np.int8)
        train['DT_W'] = ((train['DT'].dt.year - 2017) * 52 + train['DT'].dt.weekofyear).astype(np.int8)
        self.train['DT_D'] = ((train['DT'].dt.year - 2017) * 365 + train['DT'].dt.dayofyear).astype(np.int16)
        train['DT_D'] = ((train['DT'].dt.year - 2017) * 365 + train['DT'].dt.dayofyear).astype(np.int16)

        self.train['DT_hour'] = (train['DT'].dt.hour).astype(np.int8)
        self.train['DT_day_week'] = (train['DT'].dt.dayofweek).astype(np.int8)
        self.train['DT_day_month'] = (train['DT'].dt.day).astype(np.int8)

        # Possible solo feature
        train['is_december'] = train['DT'].dt.month
        self.train['is_december'] = (train['is_december'] == 12).astype(np.int8)

        # Holidays
        self.train['is_holiday'] = (train['DT'].dt.date.astype('datetime64').isin(us_holidays)).astype(np.int8)

        # test data
        self.test['DT'] = test['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
        test['DT'] = test['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
        self.test['DT_M'] = ((test['DT'].dt.year - 2017) * 12 + test['DT'].dt.month).astype(np.int8)
        test['DT_M'] = ((test['DT'].dt.year - 2017) * 12 + test['DT'].dt.month).astype(np.int8)
        self.test['DT_W'] = ((test['DT'].dt.year - 2017) * 52 + test['DT'].dt.weekofyear).astype(np.int8)
        test['DT_W'] = ((test['DT'].dt.year - 2017) * 52 + test['DT'].dt.weekofyear).astype(np.int8)
        self.test['DT_D'] = ((test['DT'].dt.year - 2017) * 365 + test['DT'].dt.dayofyear).astype(np.int16)
        test['DT_D'] = ((test['DT'].dt.year - 2017) * 365 + test['DT'].dt.dayofyear).astype(np.int16)

        self.test['DT_hour'] = (test['DT'].dt.hour).astype(np.int8)
        self.test['DT_day_week'] = (test['DT'].dt.dayofweek).astype(np.int8)
        self.test['DT_day_month'] = (test['DT'].dt.day).astype(np.int8)

        # Possible solo feature
        test['is_december'] = test['DT'].dt.month
        self.test['is_december'] = (test['is_december'] == 12).astype(np.int8)

        # Holidays
        self.test['is_holiday'] = (test['DT'].dt.date.astype('datetime64').isin(us_holidays)).astype(np.int8)

        # Total transactions per timeblock
        for col in ['DT_M', 'DT_W', 'DT_D']:
            temp_df = pd.concat([train[[col]], test[[col]]])
            fq_encode = temp_df[col].value_counts().to_dict()

            self.train[col + '_total'] = train[col].map(fq_encode)
            self.test[col + '_total'] = test[col].map(fq_encode)


class UserId(Feature):
    def create_features(self):
        # from final features. But we can use it for aggregations.
        train['uid'] = train['card1'].astype(str) + '_' + train['card2'].astype(str)
        test['uid'] = test['card1'].astype(str) + '_' + test['card2'].astype(str)

        train['uid2'] = train['uid'].astype(str) + '_' + train['card3'].astype(str) + '_' + train[
            'card5'].astype(str)
        test['uid2'] = test['uid'].astype(str) + '_' + test['card3'].astype(str) + '_' + test[
            'card5'].astype(str)

        train['uid3'] = train['uid2'].astype(str) + '_' + train['addr1'].astype(str) + '_' + train[
            'addr2'].astype(str)
        test['uid3'] = test['uid2'].astype(str) + '_' + test['addr1'].astype(str) + '_' + test[
            'addr2'].astype(str)

        train['uid4'] = train['uid3'].astype(str) + '_' + train['P_emaildomain'].astype(str)
        test['uid4'] = test['uid3'].astype(str) + '_' + test['P_emaildomain'].astype(str)

        train['uid5'] = train['uid3'].astype(str) + '_' + train['R_emaildomain'].astype(str)
        test['uid5'] = test['uid3'].astype(str) + '_' + test['R_emaildomain'].astype(str)

        for df in [train, test]:
            df['bank_type'] = df['card3'].astype(str) + '_' + df['card5'].astype(str)

        # Add values remove list
        new_columns = ['uid', 'uid2', 'uid3', 'uid4', 'uid5', 'bank_type']
        self.train[new_columns] = train[new_columns]
        self.test[new_columns] = test[new_columns]

        # Do Global frequency encoding
        def frequency_encoding(train_df, test_df, columns, self_encoding=False):
            for col in columns:
                temp_df = pd.concat([train_df[[col]], test_df[[col]]])
                fq_encode = temp_df[col].value_counts(dropna=False).to_dict()
                if self_encoding:
                    train_df[col] = train_df[col].map(fq_encode)
                    test_df[col] = test_df[col].map(fq_encode)
                else:
                    train_df[col + '_fq_enc'] = train_df[col].map(fq_encode)
                    test_df[col + '_fq_enc'] = test_df[col].map(fq_encode)
            return train_df, test_df

        i_cols = []
        train_df, test_df = frequency_encoding(train, test, new_columns, self_encoding=False)
        for n in new_columns:


            i_cols.append(n + '_fq_enc')

        self.train[i_cols] = train_df[i_cols]
        self.test[i_cols] = test_df[i_cols]


class CardTimeDistribution(Feature):
    def create_features(self):
        START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
        from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
        dates_range = pd.date_range(start='2017-10-01', end='2019-01-01')
        us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())

        # Let's add temporary "time variables" for aggregations
        # and add normal "time variables"
        for df in [train, test]:
            # Temporary variables for aggregation
            df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
            df['DT_M'] = ((df['DT'].dt.year - 2017) * 12 + df['DT'].dt.month).astype(np.int8)
            df['DT_W'] = ((df['DT'].dt.year - 2017) * 52 + df['DT'].dt.weekofyear).astype(np.int8)
            df['DT_D'] = ((df['DT'].dt.year - 2017) * 365 + df['DT'].dt.dayofyear).astype(np.int16)

            df['DT_hour'] = (df['DT'].dt.hour).astype(np.int8)
            df['DT_day_week'] = (df['DT'].dt.dayofweek).astype(np.int8)
            df['DT_day_month'] = (df['DT'].dt.day).astype(np.int8)

            # Possible solo feature
            df['is_december'] = df['DT'].dt.month
            df['is_december'] = (df['is_december'] == 12).astype(np.int8)

            # Holidays
            df['is_holiday'] = (df['DT'].dt.date.astype('datetime64').isin(us_holidays)).astype(np.int8)


        # Total transactions per timeblock
        for col in ['DT_M', 'DT_W', 'DT_D']:
            temp_df = pd.concat([train[[col]], test[[col]]])
            fq_encode = temp_df[col].value_counts().to_dict()

            train[col + '_total'] = train[col].map(fq_encode)
            test[col + '_total'] = test[col].map(fq_encode)

        for df in [train, test]:
            df['bank_type'] = df['card3'].astype(str) + '_' + df['card5'].astype(str)


        encoding_mean = {
            1: ['DT_D', 'DT_hour', '_hour_dist', 'DT_hour_mean'],
            2: ['DT_W', 'DT_day_week', '_week_day_dist', 'DT_day_week_mean'],
            3: ['DT_M', 'DT_day_month', '_month_day_dist', 'DT_day_month_mean'],
        }

        encoding_best = {
            1: ['DT_D', 'DT_hour', '_hour_dist_best', 'DT_hour_best'],
            2: ['DT_W', 'DT_day_week', '_week_day_dist_best', 'DT_day_week_best'],
            3: ['DT_M', 'DT_day_month', '_month_day_dist_best', 'DT_day_month_best'],
        }

        new_col_list = []
        for col in ['card3', 'card5', 'bank_type']:
            for df in [train, test]:
                for encode in encoding_mean:
                    encode = encoding_mean[encode].copy()
                    new_col = col + '_' + encode[0] + encode[2]
                    df[new_col] = df[col].astype(str) + '_' + df[encode[0]].astype(str)

                    temp_dict = df.groupby([new_col])[encode[1]].agg(['mean']).reset_index().rename(
                        columns={'mean': encode[3]})
                    temp_dict.index = temp_dict[new_col].values
                    temp_dict = temp_dict[encode[3]].to_dict()
                    df[new_col] = df[encode[1]] - df[new_col].map(temp_dict)
                    new_col_list.append(new_col)

                for encode in encoding_best:
                    encode = encoding_best[encode].copy()
                    new_col = col + '_' + encode[0] + encode[2]
                    df[new_col] = df[col].astype(str) + '_' + df[encode[0]].astype(str)
                    temp_dict = df.groupby([col, encode[0], encode[1]])[encode[1]].agg(['count']).reset_index().rename(
                        columns={'count': encode[3]})

                    temp_dict.sort_values(by=[col, encode[0], encode[3]], inplace=True)
                    temp_dict = temp_dict.drop_duplicates(subset=[col, encode[0]], keep='last')
                    temp_dict[new_col] = temp_dict[col].astype(str) + '_' + temp_dict[encode[0]].astype(str)
                    temp_dict.index = temp_dict[new_col].values
                    temp_dict = temp_dict[encode[1]].to_dict()
                    df[new_col] = df[encode[1]] - df[new_col].map(temp_dict)
                    new_col_list.append(new_col)

        new_col_list = list(set(new_col_list))
        self.train[new_col_list] = train[new_col_list]
        self.test[new_col_list] = test[new_col_list]


class DColumnsEngineering(Feature):
    def create_features(self):
        START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
        i_cols = ['D' + str(i) for i in range(1, 16)]

        for df in [train, test]:
            # Temporary variables for aggregation
            df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
            df['DT_M'] = ((df['DT'].dt.year - 2017) * 12 + df['DT'].dt.month).astype(np.int8)
            df['DT_W'] = ((df['DT'].dt.year - 2017) * 52 + df['DT'].dt.weekofyear).astype(np.int8)
            df['DT_D'] = ((df['DT'].dt.year - 2017) * 365 + df['DT'].dt.dayofyear).astype(np.int16)

        for df in [train, test]:
            for col in i_cols:
                df[col] = df[col].clip(0)
            # Lets transform D8 and D9 column
            # As we almost sure it has connection with hours
            df['D9_not_na'] = np.where(df['D9'].isna(), 0, 1)
            df['D8_not_same_day'] = np.where(df['D8'] >= 1, 1, 0)
            df['D8_D9_decimal_dist'] = df['D8'].fillna(0) - df['D8'].fillna(0).astype(int)
            df['D8_D9_decimal_dist'] = ((df['D8_D9_decimal_dist'] - df['D9']) ** 2) ** 0.5
            df['D8'] = df['D8'].fillna(-1).astype(int)


        periods = ['DT_D', 'DT_W', 'DT_M']
        for df in [train, test]:
            df, columns_list = values_normalization(df, periods, i_cols)


        for col in ['D1', 'D2']:
            for df in [train, test]:
                df[col + '_scaled'] = df[col] / train[col].max()


        train['uid'] = train['card1'].astype(str) + '_' + train['card2'].astype(str)
        test['uid'] = test['card1'].astype(str) + '_' + test['card2'].astype(str)

        train['uid2'] = train['uid'].astype(str) + '_' + train['card3'].astype(str) + '_' + train[
            'card5'].astype(str)
        test['uid2'] = test['uid'].astype(str) + '_' + test['card3'].astype(str) + '_' + test[
            'card5'].astype(str)

        train['uid3'] = train['uid2'].astype(str) + '_' + train['addr1'].astype(str) + '_' + train[
            'addr2'].astype(str)
        test['uid3'] = test['uid2'].astype(str) + '_' + test['addr1'].astype(str) + '_' + test[
            'addr2'].astype(str)

        train['uid4'] = train['uid3'].astype(str) + '_' + train['P_emaildomain'].astype(str)
        test['uid4'] = test['uid3'].astype(str) + '_' + test['P_emaildomain'].astype(str)

        train['uid5'] = train['uid3'].astype(str) + '_' + train['R_emaildomain'].astype(str)
        test['uid5'] = test['uid3'].astype(str) + '_' + test['R_emaildomain'].astype(str)

        train['bank_type'] = train['card3'].astype(str) + '_' + train['card5'].astype(str)
        test['bank_type'] = test['card3'].astype(str) + '_' + test['card5'].astype(str)

        i_cols = ['D' + str(i) for i in range(1, 16)]
        uids = ['uid', 'uid2', 'uid3', 'uid4', 'uid5', 'bank_type']
        aggregations = ['mean', 'std']

        ###### uIDs aggregations
        train_df, test_df, new_col_name_list = uid_aggregation(train, test, i_cols, uids, aggregations)

        new_col_name_list.extend(['D1_scaled', 'D2_scaled'])
        new_col_name_list.extend(['D9_not_na', 'D8_not_same_day', 'D8_D9_decimal_dist'])
        new_col_name_list += columns_list

        self.train[new_col_name_list] = train_df[new_col_name_list]
        self.test[new_col_name_list] = test_df[new_col_name_list]




class TransactionAmtCheck(Feature):
    def create_features(self):
        train['TransactionAmt'] = train['TransactionAmt'].clip(0, 5000)
        test['TransactionAmt'] = test['TransactionAmt'].clip(0, 5000)

        # Check if the Transaction Amount is common or not (we can use freq encoding here)
        # In our dialog with a model we are telling to trust or not to these values
        self.train['TransactionAmt_check'] = np.where(train['TransactionAmt'].isin(test['TransactionAmt']), 1, 0)
        self.test['TransactionAmt_check'] = np.where(test['TransactionAmt'].isin(train['TransactionAmt']), 1, 0)


class TransactionAmtAggregations(Feature):
    def create_features(self):
        i_cols = ['TransactionAmt']
        uids = ['card1', 'card2', 'card3', 'card5', 'uid', 'uid2', 'uid3', 'uid4', 'uid5', 'bank_type']
        aggregations = ['mean', 'std']

        train['uid'] = train['card1'].astype(str) + '_' + train['card2'].astype(str)
        test['uid'] = test['card1'].astype(str) + '_' + test['card2'].astype(str)

        train['uid2'] = train['uid'].astype(str) + '_' + train['card3'].astype(str) + '_' + train[
            'card5'].astype(str)
        test['uid2'] = test['uid'].astype(str) + '_' + test['card3'].astype(str) + '_' + test[
            'card5'].astype(str)

        train['uid3'] = train['uid2'].astype(str) + '_' + train['addr1'].astype(str) + '_' + train[
            'addr2'].astype(str)
        test['uid3'] = test['uid2'].astype(str) + '_' + test['addr1'].astype(str) + '_' + test[
            'addr2'].astype(str)

        train['uid4'] = train['uid3'].astype(str) + '_' + train['P_emaildomain'].astype(str)
        test['uid4'] = test['uid3'].astype(str) + '_' + test['P_emaildomain'].astype(str)

        train['uid5'] = train['uid3'].astype(str) + '_' + train['R_emaildomain'].astype(str)
        test['uid5'] = test['uid3'].astype(str) + '_' + test['R_emaildomain'].astype(str)

        train['bank_type'] = train['card3'].astype(str) + '_' + train['card5'].astype(str)
        test['bank_type'] = test['card3'].astype(str) + '_' + test['card5'].astype(str)

        # uIDs aggregations
        train_df, test_df, new_col_name_list = uid_aggregation(train, test, i_cols, uids, aggregations)

        periods = ['DT_D', 'DT_W', 'DT_M']
        for df in [train_df, test_df]:
            df, columns_list = values_normalization(df, periods, i_cols)

        new_col_name_list += columns_list

        self.train[new_col_name_list] = train_df[new_col_name_list]
        self.test[new_col_name_list] = test_df[new_col_name_list]



class TimeblockFrequencyEncodingBanktype(Feature):
    def create_features(self):
        START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')

        for df in [train, test]:
            # Temporary variables for aggregation
            df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
            df['DT_M'] = ((df['DT'].dt.year - 2017) * 12 + df['DT'].dt.month).astype(np.int8)
            df['DT_W'] = ((df['DT'].dt.year - 2017) * 52 + df['DT'].dt.weekofyear).astype(np.int8)
            df['DT_D'] = ((df['DT'].dt.year - 2017) * 365 + df['DT'].dt.dayofyear).astype(np.int16)


        train['uid'] = train['card1'].astype(str) + '_' + train['card2'].astype(str)
        test['uid'] = test['card1'].astype(str) + '_' + test['card2'].astype(str)

        train['uid2'] = train['uid'].astype(str) + '_' + train['card3'].astype(str) + '_' + train[
            'card5'].astype(str)
        test['uid2'] = test['uid'].astype(str) + '_' + test['card3'].astype(str) + '_' + test[
            'card5'].astype(str)

        train['uid3'] = train['uid2'].astype(str) + '_' + train['addr1'].astype(str) + '_' + train[
            'addr2'].astype(str)
        test['uid3'] = test['uid2'].astype(str) + '_' + test['addr1'].astype(str) + '_' + test[
            'addr2'].astype(str)

        train['uid4'] = train['uid3'].astype(str) + '_' + train['P_emaildomain'].astype(str)
        test['uid4'] = test['uid3'].astype(str) + '_' + test['P_emaildomain'].astype(str)

        train['uid5'] = train['uid3'].astype(str) + '_' + train['R_emaildomain'].astype(str)
        test['uid5'] = test['uid3'].astype(str) + '_' + test['R_emaildomain'].astype(str)

        train['bank_type'] = train['card3'].astype(str) + '_' + train['card5'].astype(str)
        test['bank_type'] = test['card3'].astype(str) + '_' + test['card5'].astype(str)

        for col in ['DT_M', 'DT_W', 'DT_D']:
            temp_df = pd.concat([train[[col]], test[[col]]])
            fq_encode = temp_df[col].value_counts().to_dict()

            train[col + '_total'] = train[col].map(fq_encode)
            test[col + '_total'] = test[col].map(fq_encode)


        i_cols = ['bank_type']  # ['uid','uid2','uid3','uid4','uid5','bank_type']
        periods = ['DT_M', 'DT_W', 'DT_D']

        train_df, test_df, new_col_name_list = timeblock_frequency_encoding(train, test, periods, i_cols,
                                                                            with_proportions=False,
                                                                            only_proportions=True)

        self.train[new_col_name_list] = train_df[new_col_name_list]
        self.test[new_col_name_list] = test_df[new_col_name_list]



class TimeblockFrequencyEncodingProducttype(Feature):
    def create_features(self):
        START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')

        for df in [train, test]:
            # Temporary variables for aggregation
            df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
            df['DT_M'] = ((df['DT'].dt.year - 2017) * 12 + df['DT'].dt.month).astype(np.int8)
            df['DT_W'] = ((df['DT'].dt.year - 2017) * 52 + df['DT'].dt.weekofyear).astype(np.int8)
            df['DT_D'] = ((df['DT'].dt.year - 2017) * 365 + df['DT'].dt.dayofyear).astype(np.int16)

        for col in ['DT_M', 'DT_W', 'DT_D']:
            temp_df = pd.concat([train[[col]], test[[col]]])
            fq_encode = temp_df[col].value_counts().to_dict()

            train[col + '_total'] = train[col].map(fq_encode)
            test[col + '_total'] = test[col].map(fq_encode)

        train['product_type'] = train['ProductCD'].astype(str) + '_' + train['TransactionAmt'].astype(str)
        test['product_type'] = test['ProductCD'].astype(str) + '_' + test['TransactionAmt'].astype(str)

        i_cols = ['product_type']
        periods = ['DT_D', 'DT_W', 'DT_M']
        train_df, test_df, new_col_name_list = timeblock_frequency_encoding(train, test, periods, i_cols,
                                                                            with_proportions=False,
                                                                            only_proportions=True)
        self.train[new_col_name_list] = train_df[new_col_name_list]
        self.test[new_col_name_list] = test_df[new_col_name_list]


class GroupVSumMean(Feature):
    def create_features(self):
        nans_groups = {}
        nans_df = pd.concat([train, test]).isna()

        i_cols = ['V' + str(i) for i in range(1, 340)]
        for col in i_cols:
            cur_group = nans_df[col].sum()
            if cur_group > 0:
                try:
                    nans_groups[cur_group].append(col)
                except:
                    nans_groups[cur_group] = [col]

        for n_group, n_members in nans_groups.items():
            self.train[str(n_group) + 'group_sum'] = train[n_members].to_numpy().sum(axis=1)
            self.train[str(n_group) + 'group_mean'] = train[n_members].to_numpy().mean(axis=1)

            self.test[str(n_group) + 'group_sum'] = test[n_members].to_numpy().sum(axis=1)
            self.test[str(n_group) + 'group_mean'] = test[n_members].to_numpy().mean(axis=1)


class GroupVPca(Feature):
    def create_features(self):
        nans_groups = {}
        nans_df = pd.concat([train, test]).isna()

        i_cols = ['V' + str(i) for i in range(1, 340)]
        for col in i_cols:
            cur_group = nans_df[col].sum()
            if cur_group > 0:
                try:
                    nans_groups[cur_group].append(col)
                except:
                    nans_groups[cur_group] = [col]

        for n_group, n_members in nans_groups.items():
            for col in n_members:
                sc = StandardScaler()
                sc.fit(train[[col]].fillna(0))
                train[col + '_sc'] = sc.transform(train[[col]].fillna(0))
                test[col + '_sc'] = sc.transform(test[[col]].fillna(0))

            sc_n_members = [col + '_sc' for col in n_members]

            pca = PCA(random_state=0)
            pca.fit(train[sc_n_members])
            train[sc_n_members] = pca.transform(train[sc_n_members])
            test[sc_n_members] = pca.transform(test[sc_n_members])
            self.train[sc_n_members[:5]] = train[sc_n_members[:5]]
            self.test[sc_n_members[:5]] = test[sc_n_members[:5]]


class NanGroup(Feature):
    def create_features(self):
        nans_groups = {}
        temp_df = train.isna()
        temp_df2 = test.isna()
        nans_df = pd.concat([temp_df, temp_df2])

        for col in list(nans_df):
            cur_group = nans_df[col].sum()
            if cur_group > 0:
                try:
                    nans_groups[cur_group].append(col)
                except:
                    nans_groups[cur_group] = [col]

        add_category = []
        for col in nans_groups:
            if len(nans_groups[col]) > 1:
                self.train['nan_group_' + str(col)] = np.where(temp_df[nans_groups[col]].sum(axis=1) > 0, 1, 0).astype(
                    np.int8)
                self.test['nan_group_' + str(col)] = np.where(temp_df2[nans_groups[col]].sum(axis=1) > 0, 1, 0).astype(
                    np.int8)





####### original features ##########################################################



class NumberOfNull(Feature):
    def create_features(self):
        self.train['number_of_null'] = train.isnull().sum(axis=1)
        self.test['number_of_null'] = test.isnull().sum(axis=1)


class NumberOfNullObject(Feature):
    def create_features(self):
        self.train['number_of_null_object'] = train.select_dtypes(include=object).isnull().sum(axis=1)
        self.test['number_of_null_object'] = test.select_dtypes(include=object).isnull().sum(axis=1)


class NumberOfNullNumerical(Feature):
    def create_features(self):
        self.train['number_of_null_numerical'] = train.select_dtypes(include=float).isnull().sum(axis=1)
        self.test['number_of_null_numerical'] = test.select_dtypes(include=float).isnull().sum(axis=1)


class NumberOfNullObjDivNum(Feature):
    def create_features(self):
        self.train['number_of_null_obj_div_num'] = train.select_dtypes(include=object).isnull().sum(axis=1) / train.select_dtypes(include=float).isnull().sum(axis=1)
        self.test['number_of_null_obj_div_num'] = test.select_dtypes(include=object).isnull().sum(axis=1) / test.select_dtypes(include=float).isnull().sum(axis=1)


class NumberOfNullNumDivObf(Feature):
    def create_features(self):
        self.train['number_of_null_num_div_obj'] = train.select_dtypes(include=float).isnull().sum(axis=1) / train.select_dtypes(include=object).isnull().sum(axis=1)
        self.test['number_of_null_num_div_obj'] = test.select_dtypes(include=float).isnull().sum(axis=1) / test.select_dtypes(include=object).isnull().sum(axis=1)


class NumberOfNullObjTimesNum(Feature):
    def create_features(self):
        self.train['number_of_null_obj_times_num'] = train.select_dtypes(include=object).isnull().sum(axis=1) * train.select_dtypes(include=float).isnull().sum(axis=1)
        self.test['number_of_null_obj_times_num'] = test.select_dtypes(include=object).isnull().sum(axis=1) * test.select_dtypes(include=float).isnull().sum(axis=1)

# PCAの特徴量を足してもあまり精度が上がらず... とりあえず基本的なことをやりきってから自分で試そう。
# class PCA10(Feature):
#     def create_features(self):
#         # 数値カラムの選択
#         numeric_columns = list(train.describe().columns.values)
#         numeric_columns.remove('TransactionID')
#         numeric_columns.remove('isFraud')
#         numeric_columns.remove('TransactionDT')
#         train_numeric = train[numeric_columns]
#         test_numeric = test[numeric_columns]

#         # 標準化、欠損値を最頻値で埋める
#         sc = StandardScaler()
#         train_numeric_std = pd.DataFrame(sc.fit_transform(train_numeric), columns=train_numeric.columns)
#         train_numeric_std = train_numeric_std.fillna(train_numeric_std.median())

#         # testデータは、trainデータを元に標準化と欠損値処理を行う。
#         test_numeric_std = pd.DataFrame(sc.transform(test_numeric), columns=test_numeric.columns)
#         test_numeric_std = test_numeric_std.fillna(train_numeric_std.median())

#         # PCAのインスタンス化、fitとtransform
#         pca = PCA()
#         train_pca_features = pca.fit_transform(train_numeric_std)
#         train_pca = pd.DataFrame(train_pca_features, columns=["PC{}".format(x + 1) for x in range(len(train_numeric_std.columns))])
#         test_pca_features = pca.transform(test_numeric_std)
#         test_pca = pd.DataFrame(test_pca_features, columns=["PC{}".format(x + 1) for x in range(len(test_numeric_std.columns))])

#         # 特徴量の絞り込み（PC200まで）
#         self.train[list(train_pca.columns.values[:10])] = train_pca[list(train_pca.columns.values[:10])]
#         self.test[list(train_pca.columns.values[:10])] = test_pca[list(train_pca.columns.values[:10])]

###################################################


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""


if __name__ == '__main__':
    args = get_arguments()

    folder_path = '/tmp/working/IEEE_Fraud_Detection/data/input/'

    train_identity = pd.read_csv(f'{folder_path}train_identity.csv')
    train_transaction = pd.read_csv(f'{folder_path}train_transaction.csv')
    test_identity = pd.read_csv(f'{folder_path}test_identity.csv')
    test_transaction = pd.read_csv(f'{folder_path}test_transaction.csv')

    train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
    test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')

    # train = reduce_mem_usage(train)
    # test = reduce_mem_usage(test)

    for col in ['card1']:
        valid_card = pd.concat([train[[col]], test[[col]]])
        valid_card = valid_card[col].value_counts()
        valid_card_std = valid_card.values.std()

        invalid_cards = valid_card[valid_card <= 2]
        print('Rare cards', len(invalid_cards))

        valid_card = valid_card[valid_card > 2]
        valid_card = list(valid_card.index)

        print('No intersection in Train', len(train[~train[col].isin(test[col])]))
        print('Intersection in Train', len(train[train[col].isin(test[col])]))

        train[col] = np.where(train[col].isin(test[col]), train[col], np.nan)
        test[col] = np.where(test[col].isin(train[col]), test[col], np.nan)

        train[col] = np.where(train[col].isin(valid_card), train[col], np.nan)
        test[col] = np.where(test[col].isin(valid_card), test[col], np.nan)
        print('#' * 20)

    for col in ['card2', 'card3', 'card4', 'card5', 'card6', ]:
        print('No intersection in Train', col, len(train[~train[col].isin(test[col])]))
        print('Intersection in Train', col, len(train[train[col].isin(test[col])]))

        train[col] = np.where(train[col].isin(test[col]), train[col], np.nan)
        test[col] = np.where(test[col].isin(train[col]), test[col], np.nan)
        print('#' * 20)

    del train_identity, train_transaction, test_identity, test_transaction
    gc.collect()

    generate_features(globals(), args.force)