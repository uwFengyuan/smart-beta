from influxdb import InfluxDBClient
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
import datetime as dt
import time
import datetime
import warnings
warnings.filterwarnings('ignore')
from Body import Df2Body
import requests
import json
import datetime
import os


import uqer
import warnings
import numpy as np
import pandas as pd
from influxdb import InfluxDBClient
import datetime as dt
import time
import warnings
import datetime
warnings.filterwarnings('ignore')
import datetime as dt
import time
import datetime
import uqer
import pandas as pd
from uqer import DataAPI   #优矿api  |
client = uqer.Client(token='18266a7c0ac9f8cdbe00f9b2ecb65f42316a5f78d9cc22ebabcbd923593356e4')


factor = pd.read_csv('集合竞价因子.csv',dtype = {'date':str})
del factor['Unnamed: 0']
#factor  = factor.rename(columns={'date':'tradeDate'})
print(factor.head())

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
def scale(factor,tag):
    
    factor['tradeDate'] = pd.to_datetime(factor['tradeDate'])
    factor['UTC'] = factor['tradeDate'].apply(lambda x :x-datetime.timedelta(hours=8))
    factor['UTC']=factor['UTC'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    factor['tradeDate']=factor['tradeDate'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    factor = factor.set_index([tag,'UTC'])
    return factor

factor = ticekrToStr(factor)
factor  = scale(factor,'ticker') 

print(factor.head())


Df2Body( factor ,'jhjj_FACTORS')