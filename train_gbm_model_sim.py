# -*- coding: utf-8 -*-
'''
   train gbm ensemble model
'''
import pandas as pd
import numpy as np
import sys, getopt
import logging
from os import makedirs as mkdir
import lightgbm as lgb
from keras.utils.data_utils import *
import random as rn
import math
from influxdb import InfluxDBClient
import joblib
import datetime
import uqer
from uqer import DataAPI   #优矿api
client = uqer.Client(token='18266a7c0ac9f8cdbe00f9b2ecb65f42316a5f78d9cc22ebabcbd923593356e4')
#set random seed
np.random.seed(50)
rn.seed(1234)

def set_logging(console_level, file_level, file):
    logging.basicConfig(filename=file,level=file_level)
    console = logging.StreamHandler()
    console.setLevel(console_level)
    logging.getLogger().addHandler(console)

#每日rank
def dailyrank(data,out):
    datelist = np.unique(data['date'])
    data['rank'] = 0
    for i in datelist:
        data.loc[data['date']==i,'rank'] = data.loc[data['date']==i,out].rank(ascending = True)/data.loc[data['date']==i,:].shape[0]
    return data

#每日rank
def dailydmean(data,out):
    datelist = np.unique(data['date'])
    data['pct_dmean'] = 0
    for i in datelist:
        data.loc[data['date']==i,'pct_dmean'] = data.loc[data['date']==i,out]-data.loc[data['date']==i,out].mean()
    return data

#load trainx/trainy
# traindf, out: openclose_pct1
def load_data(traindf,inddf,out,fvstart,usefv=[],pricemode='open'): 
    #calculate label
    traindf = calc_opentoclosePCT(traindf,out,pricemode)
    #mktadj pct
    #traindf = mktadj_pct(traindf,inddf,out)

    fv = [k for k in traindf.columns[fvstart:] if k in usefv]
    #print(fv)
    traindf.index = range(traindf.shape[0])
    traindf['date'] = traindf['date'].astype(int)
    traindf.loc[traindf[out].isin([np.inf,-np.inf]),out] = np.nan
    traindf = traindf.dropna(subset=[out])
    #dmean
    traindf.loc[traindf[out]>0.2,out] = 0.2
    traindf.loc[traindf[out]<-0.2,out] = -0.2

    traindf['rank'] = traindf[out]
    traindf = dailydmean(traindf,out)

    traindf = traindf.dropna(subset=['rank','pct_dmean'])
    traindf = traindf.sort_values(by=['date','ticker'])
    #drop >1/2 factor is nan
    traindf = traindf.dropna(how='all',thresh=int(len(fv)*2/3),axis=0)
    traindf = traindf.dropna(how='all',thresh=int(traindf.shape[0]*1/3),axis=1)
    #all fv
    fv = [k for k in traindf.columns[fvstart:] if k in usefv]
    trainx = traindf.loc[:,fv]
    trainy = traindf.loc[:,['ticker','date',out,'rank','pct_dmean']]
    return trainx, trainy, fv


def single_oc_clac(df, out, pricemode):
    df = df.sort_values(by='date')
    n = int(out[3:])
    df[out] = (df['close'].shift(-n) - df['open'].shift(-1)) / df['open'].shift(-1)
    return df

def calc_opentoclosePCT(wdf, out, pricemode='open'):
    wdf = wdf.groupby('ticker').apply(lambda x: single_oc_clac(x, out, pricemode))
    return wdf

def load_lrlist(lr_max,lr_min,num_rounds):
    lrlist = [lr_max+(lr_min-lr_max)*(np.log(i)/np.log(num_rounds)) for i in range(1,num_rounds+1)]
    return lrlist

def load_inddf(datespan):
    ticker = '000905'
    inddf = uqer.DataAPI.MktIdxdGet(ticker=ticker, beginDate=str(datespan[0]), endDate=str(datespan[-1]))
    inddf['date'] = [int(k.replace('-','')) for k in inddf['tradeDate']]
    inddf.index = inddf['date']
    n = int(out[3:])
    inddf[out] = (inddf['openIndex'].shift(-n-1)-inddf['openIndex'].shift(-1))/inddf['openIndex'].shift(-1)
    return inddf

def mktadj_pct(wdf,inddf,out):
    datelist = np.unique(wdf['date']).tolist()
    datelist = [k for k in datelist if k in inddf.index]
    for i in datelist:
        if i in inddf.index:
            wdf.loc[wdf['date']==i,out] = wdf.loc[wdf['date']==i,out] - inddf.loc[i,out]
        else:
            print(i,' not in index')
    return wdf

def assign_weight(trainy,weightdic):
    trainy['weight'] = 1
    trainy['absout'] = abs(trainy[out])
    wkey = [k for k in weightdic]
    wkey.sort()
    for k in wkey[:-1]:
        k2 = wkey[wkey.index(k)+1]
        trainy.loc[(trainy['absout']>=k)&(trainy['absout']<k2),'weight'] *= weightdic[k]
    k = wkey[-1]
    trainy.loc[trainy['absout']>=k,'weight'] *= weightdic[k]
    trainy['weight'] = (trainy['weight']*trainy.shape[0])/(trainy['weight'].sum())
    del(trainy['absout'])
    return trainy

def fit_base_estimator(trainx, trainy, label, params, grouplist=None):
    """Private function used to train a single base estimator."""
    if grouplist:
        lgb_train = lgb.Dataset(trainx, trainy[label].values, weight = trainy['weight'].values, group = self.grouplist)
    else:
        lgb_train = lgb.Dataset(trainx, trainy[label].values, weight = trainy['weight'].values)
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=params['num_boost_round'],
                    callbacks = [lgb.reset_parameter(learning_rate = self.params['lrlist'])]
                    )
    return gbm

if __name__ == '__main__':
    
    #default setting
    cdate = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")[:10]
    fvstart = 2
    tlen = 750
    valilen = 60
    out = 'PCT1'
    pricedir = '../model_spemlp_invfvsample/data/MktEqudAfGet/'
    weight = 'self' #mean or vali
    initweight = 'pct' #'equal' or 'pct'
    transfee = 0#0.0013
    weightdic = {0.08:3, 0.05:2, 0.01:1.5}
    n_jobs = 20
    pricemode = 'vwap'
    lr_max = 0.05
    lr_min = 0.02   
    
    #params
    try:
        opts, args = getopt.getopt(sys.argv[1:], "o:t:v:d:",["out=","cdate=","pricemode=","initweight=","valilen=","tlen="])
    except getopt.GetoptError:
        sys.exit()
    
    for opt, arg in opts:
        if opt in ("-o", "--out"):
            out = arg
            print('out: '+arg)
        elif opt in ("-d", "--cdate"):
            cdate = arg
            print('cdate: '+cdate)
        elif opt in ("-v","--valilen"):
            valilen = int(arg)
            print('valilen: ',valilen)
        elif opt in ("-t","--tlen"):
            tlen = int(arg)
            print('tlen: ',tlen)
        elif opt in ("--pricemode"):
            pricemode = arg
            print('pricemode: ',pricemode)
        elif opt in ("--initweight"):
            initweight = arg
            print('initweight: ',initweight)
        
    #dirs
    modelpath = 'model/'
    loggpath = 'logpath/'
    fvpath = 'results/'
    mkdir(modelpath, exist_ok=True)
    mkdir(loggpath, exist_ok=True)
    mkdir(fvpath, exist_ok=True)
    
    #DB client
    client = InfluxDBClient(host="175.25.50.120", port=12086, username="xtech", password="xtech123", database="factor")
    
    ##logging##
    set_logging(logging.WARNING, logging.INFO,loggpath+'_gbm'+out+'_'+cdate+'train_log')
    
    #gbm params
    params =  {
        'boosting_type' : 'gbdt',
        'num_leaves' : 31,
        'feature_fraction' : 0.8,
        'bagging_fraction' : 0.8,
        'lambda_l1':1,
        'lambda_l2':10,
        'max_bin' : 64,
        'num_boost_round': 200,
        'learning_rate' : 0.04,
        'min_data_in_leaf':10,
        'num_threads': n_jobs
    }
    params['objective'] = 'regression'
    params['metric'] = {'l2', 'auc'}
    lrlist = load_lrlist(lr_max,lr_min,params['num_boost_round'])
    params['lrlist'] = lrlist
    
    try:
        #load dataframe
        Begindate = (datetime.datetime.strptime(cdate,"%Y-%m-%d")-datetime.timedelta(days=tlen)).strftime("%Y-%m-%d")
        Enddate = (datetime.datetime.strptime(cdate,"%Y-%m-%d")+datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        datelist = dateRange(Begindate,Enddate)
        timespan = [datelist[0],datelist[-1]]
        traindf,usefv = load_timespan_traindata(client,timespan,pricedir)
        if traindf.shape[0] == 0:
            print('>>{} has no train data..%'.format(cdate))
            logging.info('>>{} has no train data..%'.format(cdate))
            sys.exit()

        #加载对冲信息
        inddf = load_inddf(timespan)

        #run train
        #load trainx/trainy valix/valiy
        trainx, trainy, fv = load_data(traindf,inddf,out,fvstart,usefv,pricemode)
        print('training data from {} to {}'.format(trainy['date'].min(),trainy['date'].max()))
        logging.info('trainning at date: {} trainwindow: {}-{}'.format(cdate,trainy['date'].min(),trainy['date'].max()))
        cdf = trainx.count()
        logging.info(cdf)
        print('usefv: ',len(fv))
        logging.info('usefv: {}'.format(len(fv)))
        wdatelist = np.unique(trainy['date']).tolist()
        wdatelist.sort()
        logging.info('train data length: {}'.format(len(wdatelist)))
        print('train data length: {}'.format(len(wdatelist)))

        trainy[out] -= transfee
        trainy = assign_weight(trainy,weightdic)
        gbmrecord.recorddf = trainy.copy()
        
        #run gbm rank: proba_0
        model = fit_base_estimator(trainx,trainy,label = 'rank',params=params)
        #save model/fv
        joblib.dump(model,modelpath+'gbm_{}_model_{}.pkl'.format(out,0))

        #run gbm pct_dmean: proba_1
        trainy['weight'] = 1
        model = fit_base_estimator(trainx,trainy,label = 'pct_dmean',params=params)
        #save model/fv
        joblib.dump(model,modelpath+'gbm_{}_model_{}.pkl'.format(out,1))
        
    except Exception as e:
        print('error occur ',e)
        logging.exception(sys.exc_info())
