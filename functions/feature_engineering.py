## Global frequency encoding
def frequency_encoding(df, columns, self_encoding=False):
    for col in columns:
        fq_encode = df[col].value_counts(dropna=False).to_dict()
        if self_encoding:
            df[col] = df[col].map(fq_encode)
        else:
            df[col+'_fq_enc'] = df[col].map(fq_encode)
    return df


def uid_aggregation(train_df, test_df, main_columns, uids, aggregations):
    new_col_name_list = []
    for main_column in main_columns:
        for col in uids:
            for agg_type in aggregations:
                new_col_name = col + '_' + main_column + '_' + agg_type
                temp_df = pd.concat([train_df[[col, main_column]], test_df[[col, main_column]]])
                temp_df = temp_df.groupby([col])[main_column].agg([agg_type]).reset_index().rename(
                    columns={agg_type: new_col_name})

                temp_df.index = list(temp_df[col])
                temp_df = temp_df[new_col_name].to_dict()

                train_df[new_col_name] = train_df[col].map(temp_df)
                test_df[new_col_name] = test_df[col].map(temp_df)
                new_col_name_list.append(new_col_name)
                print(new_col_name)
    return train_df, test_df, new_col_name_list


def values_normalization(dt_df, periods, columns):
    columns_list = []
    for period in periods:
        for col in columns:
            new_col = col + '_' + period
            dt_df[col] = dt_df[col].astype(float)

            temp_min = dt_df.groupby([period])[col].agg(['min']).reset_index()
            temp_min.index = temp_min[period].values
            temp_min = temp_min['min'].to_dict()

            temp_max = dt_df.groupby([period])[col].agg(['max']).reset_index()
            temp_max.index = temp_max[period].values
            temp_max = temp_max['max'].to_dict()

            temp_mean = dt_df.groupby([period])[col].agg(['mean']).reset_index()
            temp_mean.index = temp_mean[period].values
            temp_mean = temp_mean['mean'].to_dict()

            temp_std = dt_df.groupby([period])[col].agg(['std']).reset_index()
            temp_std.index = temp_std[period].values
            temp_std = temp_std['std'].to_dict()

            dt_df['temp_min'] = dt_df[period].map(temp_min)
            dt_df['temp_max'] = dt_df[period].map(temp_max)
            dt_df['temp_mean'] = dt_df[period].map(temp_mean)
            dt_df['temp_std'] = dt_df[period].map(temp_std)

            dt_df[new_col + '_min_max'] = (dt_df[col] - dt_df['temp_min']) / (
                    dt_df['temp_max'] - dt_df['temp_min'])
            columns_list.append(new_col + '_min_max')

            dt_df[new_col + '_std_score'] = (dt_df[col] - dt_df['temp_mean']) / (dt_df['temp_std'])
            columns_list.append(new_col + '_std_score')

            del dt_df['temp_min'], dt_df['temp_max'], dt_df['temp_mean'], dt_df['temp_std']

    return dt_df, columns_list


def timeblock_frequency_encoding(train_df, test_df, periods, columns,
                                 with_proportions=True, only_proportions=False):
    columuns_list = []
    for period in periods:
        for col in columns:
            new_col = col + '_' + period
            train_df[new_col] = train_df[col].astype(str) + '_' + train_df[period].astype(str)
            test_df[new_col] = test_df[col].astype(str) + '_' + test_df[period].astype(str)

            temp_df = pd.concat([train_df[[new_col]], test_df[[new_col]]])
            fq_encode = temp_df[new_col].value_counts().to_dict()

            train_df[new_col] = train_df[new_col].map(fq_encode)
            test_df[new_col] = test_df[new_col].map(fq_encode)

            if only_proportions:
                train_df[new_col] = train_df[new_col] / train_df[period + '_total']
                test_df[new_col] = test_df[new_col] / test_df[period + '_total']
                columuns_list.append(new_col)

            if with_proportions:
                train_df[new_col + '_proportions'] = train_df[new_col] / train_df[period + '_total']
                test_df[new_col + '_proportions'] = test_df[new_col] / test_df[period + '_total']
                columuns_list.append(new_col + '_proportions')

    return train_df, test_df, columuns_list


def get_new_columns(temp_list):
    temp_list = [col for col in list(full_df) if col not in temp_list]
    temp_list.sort()

    temp_list2 = [col if col not in remove_features else '-' for col in temp_list]
    temp_list2.sort()

    temp_list = {'New columns (including dummy)': temp_list,
                 'New Features': temp_list2}
    temp_list = pd.DataFrame.from_dict(temp_list)
    return temp_list
