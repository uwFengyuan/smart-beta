import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


# input: 十一个因子+要预测的label，output：预测结果
def function(all_factors, prediction, label):
    class MedianExtremeValueTransformer(BaseEstimator, TransformerMixin):
        def __init__(self):
            self.df_median = None
            self.df_standard = None

        def fit(self, df_columnList):
            self.df_median = df_columnList.median()
            self.df_standard = df_columnList.apply(lambda x: x - self.df_median[x.name]).abs().median()
            return self

        def scaller(self, x):
            self.di_max = self.df_median[x.name] + 5 * self.df_standard[x.name]
            self.di_min = self.df_median[x.name] - 5 * self.df_standard[x.name]
            x = x.apply(lambda v: self.di_min if v < self.di_min else v)
            x = x.apply(lambda v: self.di_max if v > self.di_max else v)
            return x

        def transform(self, df_columnList):
            df_columnList = df_columnList.apply(self.scaller)
            return df_columnList

    prediction = prediction.rename(columns={'date': 'tradeDate', 'Unnamed: 1': 'ticker', '0': 'former_pred'})
    all_factors = all_factors.drop(columns='Unnamed: 0')
    all_factors_p = all_factors.merge(prediction, on=['ticker', 'tradeDate'], how='left')
    all_factors_p = all_factors_p[~all_factors_p.former_pred.isnull()]
    all_factors_p_l = all_factors_p.merge(label, on=['ticker', 'tradeDate'], how='left')

    f_x = ['volume', 'cachgPct', 'thecommittee', 'askVolume1',
        'bidVolume1', 'caQrr', 'caTr', 'OCVP1',
        'Open/vwap-1', 'Gap', 'former_pred']
    f_y = 'openclose_pct1_rank'
    f_index = ['ticker', 'tradeDate']

    all_factors_p_l = all_factors_p_l.fillna(0)

    mevtransformer = MedianExtremeValueTransformer()
    mevtransformer.fit(all_factors_p_l[f_x])
    all_factors_p_l[f_x] = mevtransformer.transform(all_factors_p_l[f_x])

    scaler = StandardScaler()
    all_factors_p_l[f_x] = scaler.fit_transform(all_factors_p_l[f_x])


    x_train = all_factors_p_l[f_x]
    y_train = all_factors_p_l[f_y]
    print('x shape:{}, y shape: {}'.format(x_train.shape, y_train.shape))

    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)

    # 获得结果
    print('get result')
    df_result = all_factors_p_l[f_index].copy()
    df_result['y'] = y_train
    df_result['y_pred'] = y_pred
    df_result = df_result.sort_values(by=['ticker', 'tradeDate']).reset_index(drop=True)
    return df_result

prediction = pd.read_csv('pct1_cal/prediction_224_66_rolling100.csv')
all_factors = pd.read_csv('pct1_cal/集合竞价因子.csv')
label = pd.read_csv('labels/openclose_pct1.csv')
result = function(all_factors, prediction, label)
print(result)
print(result[['y', 'y_pred']].corr().iloc[0,1])
#df_result.to_csv('pct1_cal/all_factors_p_l_prediction_linear_{}.csv'.format(round(IC, 4)))
