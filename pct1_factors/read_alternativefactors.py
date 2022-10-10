# -*- coding: utf-8 -*-
"""
loading factor data from DB and price data from uqer
"""

from influxdb import InfluxDBClient
import pandas as pd
import numpy as np
import datetime
from os import makedirs as mkdir
from os.path import isfile
from os.path import join as joindir
import uqer
UQER_TOKEN = '18266a7c0ac9f8cdbe00f9b2ecb65f42316a5f78d9cc22ebabcbd923593356e4'
client = uqer.Client(token=UQER_TOKEN) 


import uqer
from uqer import DataAPI   #优矿api  
client = uqer.Client(token='18266a7c0ac9f8cdbe00f9b2ecb65f42316a5f78d9cc22ebabcbd923593356e4')

#date range
def dateRange(beginDate, endDate):  
    dates = []  
    dt = datetime.datetime.strptime(beginDate,"%Y-%m-%d")  
    date = beginDate[:]  
    while date <= endDate:  
        dates.append(date)  
        dt = dt + datetime.timedelta(1)  
        date = dt.strftime("%Y-%m-%d")
    dates.sort()
    return dates 
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
def load_timespan_factors(client,coll,timespan):
    str = "select * from "+coll+" where time >= "+'{1}{0}{1}'.format(timespan[0],"'")+" and time < "+'{1}{0}{1}'.format(timespan[1],"'")+""
    result = client.query(str)
    df = pd.DataFrame(list(result.get_points()))
    return df

def load_timespan_df(client,timespan):
    factor_coll_dic = {
                    'minute_new_FACTORS':'tradeDate','L2_FACTORS':'tradeDate','ga_FACTORS':'tradeDate','holders_FACTORS':'tradeDate',
                    'any_cor_FACTORS':'tradeDate','fund_FACTORS':'tradeDate',
                   'guba_FACTORS':'tradeDate','hk_std_FACTORS':'endDate','hkholders_FACTORS':'tradeDate',
                    'income_FACTORS':'tradeDate','jq_FACTORS':'tradeDate','liangjia_FACTORS':'tradeDate',
                   'rank_FACTORS':'tradeDate','suntime_FACTORS':'con_date',
                   'second_liangjia_FACTORS':'tradeDate','winrate_FACTORS':'tradeDate','zhaopin_FACTORS':'tradeDate',
                   'minute_FACTORS':'tradeDate','py_cov_FACTORS':'tradeDate','HK_new_FACTORS':'tradeDate',
                   'gaopin_liangjia_FACTORS':'tradeDate','jq_liangjia_FACTORS':'tradeDate',
                   'uq_liangjia_FACTORS':'tradeDate','x_tech_liangjia_FACTORS':'tradeDate','choice_FACTORS':'tradeDate'
                   }
    datalist = []
    for coll in factor_coll_dic:
        df = load_timespan_factors(client,coll,timespan)
        if df.shape[0] == 0:
            continue
        cols = [k for k in df.columns if k not in ['code','time',factor_coll_dic[coll],'tradeDate']]
        df[factor_coll_dic[coll]] = df[factor_coll_dic[coll]].replace(to_replace=[None], value=np.nan)
        df = df.dropna(subset=[factor_coll_dic[coll]])
        # print(coll,': ',cols)
        # print('data shape: ',df.shape)
        if df.shape[0] == 0:
            continue
        usetick = [k for k in df['code'] if k[0] in ['0','3','6']]
        df = df[df['code'].isin(usetick)]
        df['ticker'] = df['code'].astype(int)
        df['tradeDate'] = [k[:10] for k in df[factor_coll_dic[coll]]]
        df = df.drop_duplicates(subset=['ticker','tradeDate'],keep='last')
        df = df.set_index(['ticker','tradeDate'])
        ncols = []
        for col in cols:
            df.rename(columns={col:coll+'_'+col},inplace=True)
            ncols.append(coll+'_'+col)
        datalist.append(df[ncols])
    if len(datalist) == 0:
        return pd.DataFrame()
    wdf = pd.concat(datalist,axis=1)
    wdf = wdf.reset_index()
    wdf = wdf[wdf['tradeDate']!='NaT']
    return wdf




if __name__ == '__main__':
    #DB client
    client = InfluxDBClient(host="175.25.50.120", port=12086, username="xtech", password="xtech123", database="factor")
    #pdir = 'data/MktEqudAfGet/'
    #mkdir(pdir, exist_ok=True)
    time_b = (datetime.datetime.now()-datetime.timedelta(days=100)).strftime("%Y-%m-%d")
    time_end = datetime.datetime.now().strftime("%Y-%m-%d")
    timespan = [time_b,time_end]
    traindf = load_timespan_df(client,timespan)
    #traindf.to_csv('alternative_factors.csv')
    temp = pd.read_csv('/workspace1/liufengyuan/pct1_factors/industry_info_xtech.csv',encoding='gbk')
    ticker_list = ticekrToStr(temp)['ticker'].drop_duplicates().to_list()
    close = DataAPI.MktEqudGet(secID=u"",ticker=ticker_list,beginDate=time_b,endDate=time_end,isOpen="",field=['ticker','tradeDate','closePrice'],pandas="1")
    close['ticker'] = close['ticker'].astype(int)
    traindf['label_fake'] = 0
    
    for factor in traindf.columns:
        print(factor)
        if factor != 'ticker' and factor != 'tradeDate':
            ttt = traindf[['ticker','tradeDate',factor]]
            qqq = close.merge(ttt,how='left',on=['ticker','tradeDate'])
            qqq.set_index(['tradeDate','ticker'],inplace = True)
            final = qqq[factor].unstack()
            print(final.shape)
            final.to_csv('/workspace1/liufengyuan/pct1_factors/data/'+factor+'.csv')