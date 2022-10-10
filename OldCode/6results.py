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

"""
df_result1 = pd.DataFrame()
# 将每期的数据合并，并保存
epoch_range = range(0, 36)
for epoch in epoch_range:
    df_result = pd.read_csv('rank_{}.csv'.format(epoch))
 
    df_result['y_rank'] = df_result.groupby('tradeDate').y.rank(ascending=False) # 并列的默认使用排名均值
    df_result['y_pred_rank'] = df_result.groupby('tradeDate').y_pred.rank(ascending=False)
    # 股票数量小于等于4的合并为一个行业
    dff = df_result.groupby(['industry']).count()
    idst_map = {}# 行业序号映射字典
    for i in df_result.industry.unique():
        # 行业股票小于等于四个的 映射为0
        if i in dff[dff.ticker<=4].index.values:
            idst_map[i]=0
        # 其他的 映射为本身 相当于不用改
        else:
            idst_map[i]=i
    df_result['industry2'] = df_result.industry.map(idst_map)
    # 行业内按照 y_pred 排名，值相等的，按照出现顺序排名 first的作用
    df_result['y_rank_idst'] = df_result.groupby(['tradeDate', 'industry2']).y_pred.rank(ascending=False, method='first')
    # 分箱时使用 y_rank_idst 分箱
    df_result['class_label'] = df_result.groupby(['tradeDate', 'industry2'])['y_rank_idst'].transform(
                         lambda x: pd.qcut(x, 5, labels=range(1,6), duplicates='drop'))
    # 单期数据保存
    df_result.to_csv('new_{}.csv'.format(epoch), index=False)

    # 加入到合并数据
    df_result1 = df_result1.append(df_result)
# 合并数据保存
df_result1.to_csv('rank_new_result.csv', index=False)
"""