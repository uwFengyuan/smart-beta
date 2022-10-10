# -*- coding: utf-8 -*-
"""
集合竞价相关因子计算及存储
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
from influxdb import InfluxDBClient
from Body import Df2Body

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

def cb(factor_data1,trading_data_pretoday_date):
    stk_id=DataAPI.EquGet(ticker=u"",equTypeCD=u"",exchangeCD="",field=["secID","ticker","secShortName","exchangeCD"],pandas="1")
    secID_list=stk_id.secID.tolist()
    secID_list=list(stk_id['secID'])
    ticker_list=stk_id.ticker.tolist()
    stk_id1=stk_id[stk_id.exchangeCD=="XSHG"][['ticker','exchangeCD']]
    tickCD1= stk_id1.reset_index(drop=True)
    stk_id2=stk_id[stk_id.exchangeCD=="XSHE"][['ticker','exchangeCD']]
    tickCD2= stk_id2.reset_index(drop=True)
    #整理当日集合竞价数据
    df1 =uqer.DataAPI.SHSZTickRTIntraDayGet(ticker=tickCD1['ticker'],exchangeCD="XSHG",startTime='09:25',endTime='09:26',field=u"",pandas="1")
    df2 =uqer.DataAPI.SHSZTickRTIntraDayGet(ticker=tickCD2['ticker'],exchangeCD="XSHE",startTime='09:25',endTime='09:26',field=u"",pandas="1")
    data=[]
    data.append(df1)
    data.append(df2)
    factor_data=pd.concat(data)
    factor_data['dataDate'] = pd.to_datetime(factor_data['dataDate'])
    factor_data.drop_duplicates(subset=['dataDate','ticker'],keep='last',inplace=True)
    factor_data= factor_data.reset_index(drop=True)
    factor_data['ticker'] = factor_data['ticker'].astype(int)
    factor_data=ticekrToStr(factor_data)
    factor_data1=ticekrToStr(factor_data1)
    trading_data_pretoday_date=ticekrToStr(trading_data_pretoday_date)

    #计算前九个因子
    #去除缺失数据
    factor_data['prevClosePrice'].loc[(factor_data['prevClosePrice'] ==0)] = np.NaN
    factor_data['openPrice'].loc[(factor_data['openPrice'] ==0)] = np.NaN
    factor_data['bidVolume1'].loc[(factor_data['bidVolume1'] ==0)]=np.NaN
    factor_data=factor_data.dropna()
    #计算cachgPct和thecommittee因子
    factor_data['cachgPct']=factor_data['openPrice']/factor_data['prevClosePrice']-1
    factor_data['thecommittee']=((factor_data['bidVolume5']+factor_data['bidVolume4']+factor_data['bidVolume3']+factor_data['bidVolume2']+factor_data['bidVolume1'])-(factor_data['askVolume1']+factor_data['askVolume2']+factor_data['askVolume3']+factor_data['askVolume4']+factor_data['askVolume5']))/((factor_data['askVolume1']+factor_data['askVolume2']+factor_data['askVolume3']+factor_data['askVolume4']+factor_data['askVolume5'])+(factor_data['bidVolume5']+factor_data['bidVolume4']+factor_data['bidVolume3']+factor_data['bidVolume2']+factor_data['bidVolume1']))
    #与上一交易日所需数据合并
    factor_data = factor_data.merge(trading_data_pretoday_date[['ticker','marketValue','vwap','highestPrice','turnoverVol']],how='left',on=['ticker'])
    #计算另外四个因子
    factor_data['caTr']=(factor_data['volume']*factor_data['openPrice']*100)/factor_data['marketValue']
    factor_data['OCVP1']=factor_data['volume']/factor_data['turnoverVol']
    factor_data['Open/vwap-1']=factor_data['openPrice']/factor_data['vwap']-1
    factor_data['Gap']=factor_data['openPrice']/factor_data['highestPrice']-1
    #提取当天九个因子数据
    factor_data=factor_data[['ticker','dataDate','volume','cachgPct','thecommittee','askVolume1','bidVolume1','OCVP1','Gap','Open/vwap-1','caTr']]
    #计算caQrr因子
    factor_data1=factor_data1.rename(columns={'volume':'volume1'})
    factor_data = factor_data.merge(factor_data1[['ticker','volume1']],how='left',on=['ticker'])
    factor_data['volume1'].loc[(factor_data['volume1'] ==0)]=np.NaN
    factor_data=factor_data.dropna()
    factor_data['caQrr']=factor_data['volume']/factor_data['volume1']
    factor_data['caQrr'].loc[(factor_data['caQrr'] ==np.inf)] =1
    #得到十个因子
    factor_data=factor_data.drop(['volume1'],axis=1)
    str=trading_data_pretoday_date['tradeDate'][1]
    factor_data['dataDate']=str
    factor_data= factor_data.reset_index(drop=True)
    return factor_data,str
def scale(factor,tag):
    factor = factor.rename(columns={'dataDate':'tradeDate'})
    factor['tradeDate'] = pd.to_datetime(factor['tradeDate'])
    factor['UTC'] = factor['tradeDate'].apply(lambda x :x-datetime.timedelta(hours=8))
    factor['UTC']=factor['UTC'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    factor['tradeDate']=factor['tradeDate'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    factor = factor.set_index([tag,'UTC'])
    return factor
def main():
    #os.chdir('/home/zhangjingxuan/YYN')
    factor_data1 = pd.read_csv('/workspace1/liufengyuan/pct1_factors/jhjj_data/上一交易日竞价数据.csv')
    trading_data_pretoday_date=pd.read_csv('/workspace1/liufengyuan/pct1_factors/jhjj_data/上一交易日行情数据.csv')
    factor_data,str=cb(factor_data1,trading_data_pretoday_date)
    factor_data.to_csv('/workspace1/liufengyuan/pct1_factors/jhjj_data/jhjj_factor_'+str+'.csv')
    print(factor_data)
    factor  = scale(factor_data,'ticker')
    print(factor)
    Df2Body( factor ,'jhjj_FACTORS')
 
if __name__ == "__main__":
    main()