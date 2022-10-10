import os
import time
import joblib
import pickle
import datetime
import numpy as np
import pandas as pd
from influxdb import InfluxDBClient
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
pd.options.mode.chained_assignment = None

# check missing values
def check_missing_values(origin, new):
    print('The number of factors of my MODEL: ', len(origin))
    count1 = 0
    for file in origin:
        if file not in new:
            print(file)
            count1 += 1
    print('缺失值', count1)

    print('The number of factors in my DATABASE for each day: ', len(new))
    count2 = 0
    for file in new:
        if file not in origin:
            print(file)
            count2 += 1
    print('多余值', count2)

    return (count1 == 0 and count2 == 0)

# reduce memory usage
def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

"""
# convert ticker to string type
def ticekrToStr(job_sum):
    job_sum = job_sum[job_sum.ticker<700000]
    job_sum.loc[job_sum.ticker<10,'temp']='00000'
    job_sum.loc[(job_sum.ticker<100)&(job_sum.ticker>=10),'temp']='0000'
    job_sum.loc[(job_sum.ticker<1000)&(job_sum.ticker>=100),'temp']='000'
    job_sum.loc[(job_sum.ticker<10000)&(job_sum.ticker>=1000),'temp']='00'
    job_sum.loc[job_sum.temp==job_sum.temp,'ticker'] = job_sum[job_sum.temp==job_sum.temp]['temp']+job_sum[job_sum.temp==job_sum.temp]['ticker'].astype(str)
    del job_sum['temp']
    job_sum['ticker'] = job_sum['ticker'].astype(str)
    return job_sum

# read and transform each factor
def each_factor(file_loc, name, date):
    alpha = pd.read_csv(file_loc + name + '.csv')
    if alpha.columns.values.tolist()[0] == 'date':
        alpha = alpha.rename(columns={"date": "tradeDate"})
    alpha = alpha.set_index(['tradeDate']).stack().reset_index().rename(columns={'level_1': 'ticker', 0: name})
    alpha['ticker'] = alpha['ticker'].astype(int)
    if date != '':
        alpha = alpha[alpha.tradeDate == date]
    alpha = alpha.set_index(['ticker', 'tradeDate'])
    alpha = alpha.sort_index(ascending=True)
    return alpha

# combine all factors
def combine_factors(file_loc, f_x, date, thresh):        
    parallel_obj = Parallel(n_jobs=-1)(delayed(each_factor)(file_loc, name, date) for name in f_x)
    result = pd.concat(parallel_obj, axis=1)
    result = result.reset_index()
    result = result.sort_values(by=['ticker', 'tradeDate']).reset_index(drop=True)
    result = result.dropna(thresh=thresh)
    return result
"""
# exclude extreme values
class ExcludeExtreme(BaseEstimator, TransformerMixin):
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

# deal with factors
def run_df(f_x_221, date, df):
    print('----- DATE {}------'.format(date))
    pd.options.mode.chained_assignment = None
    df2 = df[df.tradeDate == date]

    mevtransformer = ExcludeExtreme()
    mevtransformer.fit(df2[f_x_221])
    df2[f_x_221] = mevtransformer.transform(df2[f_x_221])

    df2 = df2[f_index + f_x_221]
    return df2

# get factors from a database
def get_factors(date, select_date):
    
    table_name = 'all_Factors'
    client = InfluxDBClient(host="175.25.50.116", port=12086, username="xtech", password="xtech2022", database="factor")
    result = client.query("select * from /"+ table_name +"/ where time >= '{}' ".format(select_date))
    result = pd.DataFrame(list(result.get_points()))

    if result.empty:
        print('We haven\'t gotten the needed factors data.')
        os._exit(0)
    
    factors = result.drop(columns= 'time')
    factors = factors.rename(columns= {'code': 'ticker'})
    factors['ticker'] = factors['ticker'].astype(int)
    factors.insert(0, 'tradeDate', factors.pop('tradeDate'))
    factors.insert(0, 'ticker', factors.pop('ticker'))
    factors['tradeDate'] = factors.tradeDate.apply(lambda x: x[:-9])
    factors = factors.dropna(thresh=70)
    if 'ga_FACTORS_sfactor175 ' in factors.columns.to_list():
        print('Input factor name \'ga_FACTORS_sfactor175 \' should be modified.')
        factors = factors.rename(columns={'ga_FACTORS_sfactor175 ': 'ga_FACTORS_sfactor175'})
    else:
        print('There is no factor \'ga_FACTORS_sfactor175\'')
        factors['ga_FACTORS_sfactor175'] = np.NAN
        factors['choice_FACTORS_HQFW_SIGNAL'] = np.NAN
        factors['choice_FACTORS_NEWS_SIGNAL'] = np.NAN
    #print(factors)
    factors = factors[factors.tradeDate == date]
    #factors.fillna(value=np.nan, inplace=True)
    """
    #读取昨天的普通因子
    old_factors = combine_factors('pct1_cal/每日预测/data/', f_x_221, date, 67)
   
    #读取昨天的集合竞价因子
    jiheyingzi = pd.read_csv('pct1_cal/每日预测/jhjj_data/jhjj_factor_{}.csv'.format(date))
    jiheyingzi= jiheyingzi.drop(columns='Unnamed: 0')
    jiheyingzi = jiheyingzi.rename(columns={"dataDate": "tradeDate"})
    old_factors = old_factors.merge(jiheyingzi, on=['ticker', 'tradeDate'], how='left')
    
    tickers = old_factors.ticker.unique().tolist()
    print(len(tickers))
    factors = factors[factors['ticker'].isin(tickers)]
    print(factors)
    """
    return factors

# get the prediction result
def get_result(f_x, file_location, date, select_date):
    factors = get_factors(date, select_date)
    new_factors = factors.columns.tolist()
    new_factors.remove('tradeDate')
    new_factors.remove('ticker')
    all_here = check_missing_values(f_x, new_factors)
    
    if not all_here:
        print('Some factors are missing.')
        os._exit(0)

    try:
        # 与训练数据保持一致，都是只去极值
        factors = run_df(f_x, date, factors)
        # 读取只去极值的训练model，并且用来预测
        model = joblib.load(file_location + '/train_model_new.m')
        y_pred = model.predict(factors[f_x])

        #整理预测结果
        result = factors[f_index]
        result['prediction'] = y_pred
        result.rename(columns = {'tradeDate': 'date'}, inplace = True)
        result['date'] = result.date.apply(lambda x: str(x).replace('-', ''))
        result = result.sort_values(by=['prediction'], ascending= False).reset_index(drop=True)
        return result
    except TypeError:
        result = factors.dtypes
        result.drop('tradeDate', inplace = True)
        empty_factors = pd.DataFrame(result[result == object])
        empty_factors.reset_index(inplace=True)
        empty_factors.rename(columns={0: 'type', 'index': 'name'}, inplace=True)
        empty_factors['Empty_fraction'] = empty_factors.name.apply(lambda x: 
            (factors[x].isnull().sum() / factors[x].shape[0]))
        empty_factors.set_index('name', inplace=True)
        print('Empty factors: \n{}'.format(empty_factors))
        print('Some factors are empty.')
        os._exit(0)

#计算耗时
start = time.time()

# 参数设置
file_location = '/data/liufengyuan/pct1_cal/dailyprediction'
f_index = ['ticker', 'tradeDate']
f_x = pickle.load(open(file_location + '/f_x_221', 'rb'))
extra_factors = ['volume', 'cachgPct', 'thecommittee', 'askVolume1',
       'bidVolume1', 'caQrr', 'caTr', 'OCVP1',
       'Open/vwap-1', 'Gap']
extra_factors = ['jhjj_FACTORS_' + x for x in extra_factors]
f_x.extend(extra_factors)

#自动计算要预测的日期
prediction_date = datetime.date.today()
if prediction_date.isoweekday() == 1:
    prediction_date -= datetime.timedelta(days= 3)
elif prediction_date.isoweekday() in set((2, 3, 4, 5)):
    prediction_date -= datetime.timedelta(days= 1)
else:
    prediction_date -= datetime.timedelta(days= prediction_date.isoweekday() % 5)
#微调我们想要的日期
prediction_date -= datetime.timedelta(days= 7)

date = prediction_date.strftime("%Y-%m-%d")
store_date = prediction_date.strftime("%Y%m%d")
select_date = (prediction_date - datetime.timedelta(days= 1)).strftime("%Y-%m-%d")
print('Prediciton factors: ', date)
print('Store location: ', store_date)
print('Loading data date: ', select_date)

#得到预测结果并存储
result = get_result(f_x, file_location, date, select_date)
#if result is None:
#    print('Stop predicting! The predicted result will be biased since some factors are empty.')
#else:
print('Our prediction is: ')
result.to_csv(file_location + '/prediction/{}.csv'.format(store_date), index = False)
print(result)

#计算耗时
print(f'耗时:{time.time() - start}')