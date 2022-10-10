# -*- coding: utf-8 -*-
"""
集合竞价前期数据计算及存储
"""
import os
import tqdm
import uqer 
from uqer import DataAPI   #优矿api   
import pandas as pd
import numpy as np
import datetime as dt
import time
import datetime
import statsmodels.api as sm

import warnings
warnings.filterwarnings('ignore')
# 连接优矿数据库
client = uqer.Client(token='18266a7c0ac9f8cdbe00f9b2ecb65f42316a5f78d9cc22ebabcbd923593356e4')

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
def preReadyData1():
    stk_id=DataAPI.EquGet(ticker=u"",equTypeCD=u"",exchangeCD="",field=["secID","ticker","secShortName","exchangeCD"],pandas="1")
    secID_list=stk_id.secID.tolist()
    secID_list=list(stk_id['secID'])
    ticker_list=stk_id.ticker.tolist()
    stk_id1=stk_id[stk_id.exchangeCD=="XSHG"][['ticker','exchangeCD']]
    tickCD1= stk_id1.reset_index(drop=True)
    stk_id2=stk_id[stk_id.exchangeCD=="XSHE"][['ticker','exchangeCD']]
    tickCD2= stk_id2.reset_index(drop=True)

    #上一个交易日集合竞价数据
    #初始化定义
    today = time.strftime("%Y-%m-%d")
    today_date = time.strftime("%Y%m%d")
    date=DataAPI.TradeCalGet(exchangeCD=u"XSHG,XSHE",beginDate=u"20190301",endDate=today_date,isOpen=u"1",field=u"",pandas="1")
    pretoday_date=date['prevTradeDate'][len(date['prevTradeDate'])-1]
    #整理数据
    data1=[]
    for i in range(len(tickCD1['ticker'])):
        df3=DataAPI.SHSZTicksHistOneDay2Get(ticker=tickCD1['ticker'][i],exchangeCD=u"XSHG",tradeDate=pretoday_date,field="",pandas="1")
        data1.append(df3)
    for i in range(len(tickCD2['ticker'])):
        df4=DataAPI.SHSZTicksHistOneDay2Get(ticker=tickCD2['ticker'][i],exchangeCD=u"XSHE",tradeDate=pretoday_date,field="",pandas="1")
        data1.append(df4)
    factor_data1=pd.concat(data1)
    factor_data1['dataTime'] = pd.to_datetime(factor_data1['dataTime'])
    start_time=datetime.datetime.strptime('09:25:00','%H:%M:%S').time()
    end_time=datetime.datetime.strptime('09:26:00','%H:%M:%S').time()
    factor_data1=factor_data1[(factor_data1.dataTime.dt.time>=start_time) & (factor_data1.dataTime.dt.time<=end_time)]
    factor_data1['tradeDate'] = pd.to_datetime(factor_data1['tradeDate'])
    factor_data1.drop_duplicates(subset=['tradeDate','ticker'],keep='last',inplace=True)
    factor_data1= factor_data1.reset_index(drop=True)
    factor_data1['ticker'] = factor_data1['ticker'].astype(int)
    factor_data1=ticekrToStr(factor_data1)
    return factor_data1
    
def preReadyData2():
    stk_id=DataAPI.EquGet(ticker=u"",equTypeCD=u"",exchangeCD="",field=["secID","ticker","secShortName","exchangeCD"],pandas="1")
    secID_list=stk_id.secID.tolist()
    secID_list=list(stk_id['secID'])
    ticker_list=stk_id.ticker.tolist()
    stk_id1=stk_id[stk_id.exchangeCD=="XSHG"][['ticker','exchangeCD']]
    tickCD1= stk_id1.reset_index(drop=True)
    stk_id2=stk_id[stk_id.exchangeCD=="XSHE"][['ticker','exchangeCD']]
    tickCD2= stk_id2.reset_index(drop=True)

    #上一个交易日行情数据数据
    #初始化定义
    today = time.strftime("%Y-%m-%d")
    today_date = time.strftime("%Y%m%d")
    date=DataAPI.TradeCalGet(exchangeCD=u"XSHG,XSHE",beginDate=u"20190301",endDate=today_date,isOpen=u"1",field=u"",pandas="1")
    pretoday_date=date['prevTradeDate'][len(date['prevTradeDate'])-1]
    #上一交易日行情数据导入
    trading_data_pretoday_date=DataAPI.MktEqudGet(ticker=ticker_list,beginDate=pretoday_date,endDate=pretoday_date,isOpen="1",field=u"",pandas="1")
    trading_data_pretoday_date['tradeDate'] = pd.to_datetime(trading_data_pretoday_date['tradeDate'])
    trading_data_pretoday_date['ticker'] = trading_data_pretoday_date['ticker'].astype(int)
    trading_data_pretoday_date=ticekrToStr(trading_data_pretoday_date)
    trading_data_pretoday_date['vwap'].loc[(trading_data_pretoday_date['vwap'] ==0)] = np.NaN
    trading_data_pretoday_date['openPrice'].loc[(trading_data_pretoday_date['openPrice'] ==0)] = np.NaN
    trading_data_pretoday_date=trading_data_pretoday_date.dropna()
    trading_data_pretoday_date=trading_data_pretoday_date.reset_index(drop=True)
    return trading_data_pretoday_date

def main():
    factor_data1= preReadyData1()
    trading_data_pretoday_date= preReadyData2()
    #os.chdir('/home/zhangjingxuan/YYN')
    factor_data1.to_csv('/workspace1/liufengyuan/pct1_factors/jhjj_data/上一交易日竞价数据.csv')
    trading_data_pretoday_date.to_csv('/workspace1/liufengyuan/pct1_factors/jhjj_data/上一交易日行情数据.csv')

if __name__ == "__main__":
    main()