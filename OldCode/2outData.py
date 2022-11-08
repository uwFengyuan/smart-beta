import  numpy as np
import pandas as pd
import uqer
from joblib import Parallel, delayed
from uqer import DataAPI   #优矿api
client = uqer.Client(token='18266a7c0ac9f8cdbe00f9b2ecb65f42316a5f78d9cc22ebabcbd923593356e4')
from pandas.core.common import SettingWithCopyWarning
# a list of functions
## convert ticker numbers into strings
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

def get_target(ticker_name):
    # get close price and market value
    outData = DataAPI.MktEqudGet(ticker=ticker_name, beginDate='2017-07-25', endDate='2022-04-05', pandas="1")
    outData = outData[['ticker', 'tradeDate', 'closePrice', 'marketValue']]
    # get industry
    industry_data = DataAPI.EquIndustryGet(ticker=ticker_name, pandas="1")
    industry_name = industry_data.at[0, 'industryName1']
    # get listDate
    list_date = DataAPI.EquGet(ticker=ticker_name ,pandas="1")
    list_date = list_date.at[0, 'listDate']
    # combine data
    outData['industry'] = industry_name
    outData['listData'] = list_date
    outData = outData.sort_values(by=['tradeDate']).reset_index(drop=True)
    return outData[['ticker', 'tradeDate', 'marketValue', 'industry', 'listData']]

print('load data')
data = pd.read_csv('/data/liufengyuan/pct1_cal/1alter_alphas_066.csv', usecols=['ticker'])
print(data)
strData = ticekrToStr(data)
tickers = strData.ticker.unique()
tickers = tickers.tolist()

df_merge = get_target(tickers)
print('sort values')
df_merge = df_merge.sort_values(by=['ticker', 'tradeDate']).reset_index(drop=True)
print(df_merge)
#df_merge.to_csv('/data/liufengyuan/pct1_cal/3uqer_data.csv', index = False)

df_merge.industry = df_merge.industry.fillna('其他')
industry_to_number = {}
for i, v in enumerate(df_merge.industry.sort_values().unique()):
    industry_to_number[v] = i+1
print(industry_to_number)
df_merge.industry = df_merge.industry.map(industry_to_number)
df_idst = pd.get_dummies(df_merge.industry, prefix='idst')
df = df_merge.merge(df_idst, left_index=True, right_index=True)
df['log_marketValue'] = df.marketValue.apply(np.log)
#df.to_csv('/data/liufengyuan/pct1_cal/4uqer_idst_log.csv', index = False)