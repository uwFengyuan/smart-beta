import glob

import pandas as pd
from joblib import Parallel, delayed
import time
"""
# result = pd.read_csv('rank_new_result.csv')
# print(result.head(50))
result_time = pd.read_csv('rank_new_result.csv') # result[result.tradeDate == '2019-01-02']
print(result_time)
result_time = result_time.sort_values(by=['tradeDate']).reset_index(drop=True)
df_result = pd.DataFrame({'ticker':result_time.ticker, 'tradeDate':result_time.tradeDate,
                          'y':result_time.y, 'y_pred':result_time.y_pred})
out = df_result[['y', 'y_pred']].corr().iloc[0,1]
print(out)
result_time['y_rank'] = result_time.y.rank(ascending=False) # 并列的默认使用排名均值
result_time['y_pred_rank'] = result_time.y_pred.rank(ascending=False)
out = result_time[['y_rank', 'y_pred_rank']].corr().iloc[0,1]
print(out)
"""

filenames = glob.glob('pct1_cal/all_factors_p_l_prediction_linear.csv')
for file in filenames:
    #name = file[12:-21]
    #if name == "UQER_LGBMRegressor-PCT5":
    #    name = "UQER_LGBMRegressor-PCT5_222_036"
    #name = name.replace('LGBMRegressor-', 'Prediction_')
    #print('load {} data'.format(name))
    name = file[28: -6]
    print(name)
    df = pd.read_csv(file)
    df.rename(columns = {'tradeDate': 'date', 'y_pred': 'prediction'}, inplace = True)
    df['date'] = df.date.apply(lambda x: str(x).replace('-', ''))
    df = df[['ticker', 'date', 'prediction']]

    dates = df.date.sort_values().unique()
    def run(date):
        print(date)
        temp = df[df.date == date]
        temp = temp.sort_values(by=['prediction'], ascending= False).reset_index(drop=True)
        #temp.to_csv('Predictions/zz500_{}/{}.csv'.format(name, date), index=False)
        temp.to_csv('pct1_cal/linear_prediction/{}.csv'.format(date), index=False)

    print('PARALLEL')
    start = time.time()
    parallel_obj = Parallel(n_jobs=48)(delayed(run)(date) for date in dates)
    print(f'耗时:{time.time() - start}')