import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import datetime
import glob


# input: 十个因子+预测值，output：组合结果
def function(all_factors, prediction, prediction_date):
    all_factors = all_factors.rename(columns={'dataDate': 'date'})
    all_factors = all_factors.drop(columns='Unnamed: 0')
    all_factors['date'] = all_factors.date.apply(lambda x: str(x).replace('-', ''))

    prediction = prediction.drop(columns='Unnamed: 0')
    prediction['ticker'] = prediction['ticker'] .astype(np.int64)
    prediction['date'] = prediction.date.apply(lambda x: str(x))

    all_factors_p = all_factors.merge(prediction, on=['ticker', 'date'], how='left')
    all_factors_p = all_factors_p[~all_factors_p.prediction.isnull()]

    f_x = ['volume', 'cachgPct', 'thecommittee', 'askVolume1',
        'bidVolume1', 'caQrr', 'caTr', 'OCVP1', 'Open/vwap-1', 'Gap']
    f_index = ['ticker', 'date']

    scaler = StandardScaler()
    all_factors_p[f_x + ['prediction']] = scaler.fit_transform(all_factors_p[f_x + ['prediction']])

    result = all_factors_p[f_index]
    result['prediction'] = all_factors_p[f_x].sum(axis=1)/20 + all_factors_p['prediction']/2
    result = result.sort_values(by=['ticker', 'date']).reset_index(drop=True)

    result.to_csv('linear_combination_result/{}.csv'.format(prediction_date.strftime("%Y%m%d")), index=False)
    print(result)

prediction_date = datetime.date.today()
if prediction_date.isoweekday() == 1:
    prediction_date -= datetime.timedelta(days= 3)
elif prediction_date.isoweekday() in set((2, 3, 4, 5)):
    prediction_date -= datetime.timedelta(days= 1)
else:
    prediction_date -= datetime.timedelta(days= prediction_date.isoweekday() % 5)
print('Prediciton date: ', prediction_date)

filename1 = 'jhjj_data/jhjj_factor_' + prediction_date.strftime("%Y-%m-%d") + '.csv'
filename2 = 'prediction_results/' + prediction_date.strftime("%Y%m%d") + '.csv'
print('Factors\' file location: ', filename1)
print('Signal\' file location: ', filename2)

all_factors = pd.read_csv(filename1)
prediction = pd.read_csv(filename2)
result = function(all_factors, prediction, prediction_date)
