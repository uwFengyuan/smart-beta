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
    time_end = (datetime.datetime.now()-datetime.timedelta(days=1)).strftime("%Y-%m-%d")
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



from utils import load_data,  load_industry_info
from concurrent.futures import ProcessPoolExecutor
from alpha_calculation import Alpha_Calculation
from get_formula import formula_to_tree, tree_to_formula
import traceback
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int)
parser.add_argument("--threshold", type=float)
args = parser.parse_args()





def calculation(tree):
    try:
        if isinstance(tree, str):
            tree = formula_to_tree(tree, var_names)
        Calculator = Alpha_Calculation()
        Calculator.set_industry_info(ind_info)
        Calculator.calculate_features(tree, data, var_names)
        # tree.data = tree.data.rank(axis=1, numeric_only=True, ascending=True, pct=True)
        tree.do_partial_deletion(depth=1)
        return tree
    except Exception as e:
        if isinstance(tree, str):
            print(tree)
        else:
            print(tree_to_formula(tree))
        print(traceback.format_exc())


if __name__ == '__main__':
    # zz1000 = pd.read_csv('/home/zhangtianping/AutoAlpha3/MS/DataManagement/zz1000_cons/2020-12-31.csv', encoding="gbk")
    # zz1000 = zz1000['ticker'].to_list()

    data = load_data('/workspace1/liufengyuan/pct1_factors/data/',label_name = 'label_fake')
    print(data)
    label = data['label']
    ind_info = load_industry_info(label, path='/workspace1/liufengyuan/pct1_factors/industry_info_xtech.csv', index_col='ticker')
    all_index = label.stack(dropna=False).index
    del data['label']
    var_names = list(data.keys())
    population = []


    new_features = ['max(var((gaopin_liangjia_FACTORS_c19*minute_FACTORS_RSJ),15),sigmoid(sub_rank(gaopin_liangjia_FACTORS_c79,gaopin_liangjia_FACTORS_c78)))', 'rank(((uq_liangjia_FACTORS_ACD6/gaopin_liangjia_FACTORS_c71)+wma(winrate_FACTORS_prob,15)))', 'lowday(relu(max(uq_liangjia_FACTORS_LCAP,gaopin_liangjia_FACTORS_c20)),15)', 'rank(lowday(tsmin(liangjia_FACTORS_BIAS60,20),30))', 'mean(ind_rank(sigmoid(choice_FACTORS_HQFW_SIGNAL)),25)', 'delay(kurtosis(highday(fund_FACTORS_num,25),20),15)', 'rank(wma(sign(minute_new_FACTORS_ILLIQ),20))', 'rank(tsmin((second_liangjia_FACTORS_lag-second_liangjia_FACTORS_lag),15))', 'sum(ind_rank(div_rank(choice_FACTORS_HQFW_SIGNAL,choice_FACTORS_HQFW_SIGNAL)),30)', 'rank(tsrank(sign(any_cor_FACTORS_mom_cl),25))', 'min(std(covariance(gaopin_liangjia_FACTORS_c125,gaopin_liangjia_FACTORS_c78,20),20),sub_rank(abs(gaopin_liangjia_FACTORS_c80),gaopin_liangjia_FACTORS_c77))', '(lowday(sqrt(uq_liangjia_FACTORS_LCAP),15)*sum(div_rank(uq_liangjia_FACTORS_LFLO,minute_FACTORS_adj_mom),20))', 'rank(min(ind_rank(second_liangjia_FACTORS_lag),gaopin_liangjia_FACTORS_c22))', 'ind_neutralize(sigmoid(delay(gaopin_liangjia_FACTORS_c13,20)))', 'rank((ind_rank(minute_FACTORS_weipan)*(gaopin_liangjia_FACTORS_c20+gaopin_liangjia_FACTORS_c22)))', '(min(std(gaopin_liangjia_FACTORS_c14,25),square(gaopin_liangjia_FACTORS_c125))/(sqrt(jq_liangjia_FACTORS_VSTD10)*highday(uq_liangjia_FACTORS_LCAP,25)))', '(decaylinear(sqrt(winrate_FACTORS_prob),30)*minute_FACTORS_arpp)', 'rank(tsrank(max(gaopin_liangjia_FACTORS_c69,uq_liangjia_FACTORS_LFLO),25))', 'rank(sigmoid((uq_liangjia_FACTORS_MTM*gaopin_liangjia_FACTORS_c30)))', '(lowday(tsmin(uq_liangjia_FACTORS_LCAP,15),30)/square(var(minute_new_FACTORS_TMA_turn,30)))', 'tsrank(sub_rank((gaopin_liangjia_FACTORS_c79+gaopin_liangjia_FACTORS_c14),max(minute_new_FACTORS_TMA_turn,gaopin_liangjia_FACTORS_c77)),30)', '(covariance(tsmax(gaopin_liangjia_FACTORS_c80,15),delta(gaopin_liangjia_FACTORS_c77,20),15)+tsrank(max(gaopin_liangjia_FACTORS_c71,gaopin_liangjia_FACTORS_c125),15))', 'div_rank((sum(gaopin_liangjia_FACTORS_c80,20)/gaopin_liangjia_FACTORS_c20),median(var(winrate_FACTORS_prob,20),30))', 'delta(relu(sub_rank(gaopin_liangjia_FACTORS_c77,gaopin_liangjia_FACTORS_c107)),30)', 'rank(tsrank(sign(gaopin_liangjia_FACTORS_c22),20))', 'rank(tsmin(sign(gaopin_liangjia_FACTORS_c30),15))', '(min(sub_rank(gaopin_liangjia_FACTORS_c71,gaopin_liangjia_FACTORS_c69),tsmax(any_cor_FACTORS_mom_cl,20))+tsrank(sqrt(uq_liangjia_FACTORS_LCAP),15))', 'rank(std(sign(gaopin_liangjia_FACTORS_c69),20))', 'min(square(tsrank(gaopin_liangjia_FACTORS_c107,30)),min((gaopin_liangjia_FACTORS_c77-gaopin_liangjia_FACTORS_c80),var(choice_FACTORS_HQFW_SIGNAL,20)))', '(tsrank(sqrt(gaopin_liangjia_FACTORS_c71),15)+min(square(gaopin_liangjia_FACTORS_c78),rank(jq_liangjia_FACTORS_boll_down)))', 'rank(sigmoid((L2_FACTORS_Stren_42/minute_new_FACTORS_TMA_turn)))', 'rank(delta(sum(winrate_FACTORS_prob,20),15))', 'ind_neutralize(((uq_liangjia_FACTORS_LCAP/x_tech_liangjia_FACTORS_factor_model_genetic_028)-tsmin(gaopin_liangjia_FACTORS_c22,20)))', '(relu((gaopin_liangjia_FACTORS_c20/gaopin_liangjia_FACTORS_c79))*winrate_FACTORS_prob)', '(var((liangjia_FACTORS_BIAS60-second_liangjia_FACTORS_bear),20)*((gaopin_liangjia_FACTORS_c78+liangjia_FACTORS_BIAS60)+skewness(gaopin_liangjia_FACTORS_c120,30)))', 'max(abs(lowday(fund_FACTORS_num,15)),min(delay(suntime_FACTORS_market_confidence_5d,25),sum(minute_new_FACTORS_ILLIQ,20)))', '(sign((minute_new_FACTORS_IMI+uq_liangjia_FACTORS_ACD6))*sqrt(max(winrate_FACTORS_prob,minute_FACTORS_arpp)))', 'abs(covariance(sigmoid(uq_liangjia_FACTORS_LFLO),uq_liangjia_FACTORS_LFLO,20))', 'ind_rank(sign(tsmax(minute_new_FACTORS_TMA_turn,20)))', 'rank(tsmin(tsmin(gaopin_liangjia_FACTORS_c71,25),20))', 'ind_rank(sign(abs(liangjia_FACTORS_Rank1M)))', 'rank(square(tsrank(gaopin_liangjia_FACTORS_c22,20)))', 'ind_rank(var(sign(choice_FACTORS_HQFW_SIGNAL),20))', 'rank(lowday(std(fund_FACTORS_num,25),15))', '(sub_rank(sign(x_tech_liangjia_FACTORS_factor_model_genetic_028),sign(gaopin_liangjia_FACTORS_c125))/sign(max(gaopin_liangjia_FACTORS_c120,minute_FACTORS_arpp)))', 'std(lowday(max(fund_FACTORS_num,uq_liangjia_FACTORS_MTM),15),15)', 'std(sub_rank(sigmoid(gaopin_liangjia_FACTORS_c14),sign(gaopin_liangjia_FACTORS_c20)),25)', 'ind_rank(lowday(sign(choice_FACTORS_HQFW_SIGNAL),30))', 'sqrt(div_rank(sign(uq_liangjia_FACTORS_MTM),(x_tech_liangjia_FACTORS_ADX/x_tech_liangjia_FACTORS_ADX)))', '(sum(tsmax(gaopin_liangjia_FACTORS_c13,20),20)*((liangjia_FACTORS_BIAS60+uq_liangjia_FACTORS_ACD6)*std(any_cor_FACTORS_mom_cl,20)))', 'sqrt((square(gaopin_liangjia_FACTORS_c14)/median(gaopin_liangjia_FACTORS_c71,30)))', 'decaylinear(rank(sign(gaopin_liangjia_FACTORS_c78)),25)', 'ind_rank(mean(sign(gaopin_liangjia_FACTORS_c13),30))', 'square(max(div_rank(gaopin_liangjia_FACTORS_c69,gaopin_liangjia_FACTORS_c71),(liangjia_FACTORS_Rank1M+L2_FACTORS_f2)))', 'div_rank(tsrank(tsmax(uq_liangjia_FACTORS_LFLO,20),30),tsmin(highday(gaopin_liangjia_FACTORS_c79,15),30))', 'max((sigmoid(uq_liangjia_FACTORS_LFLO)+tsrank(gaopin_liangjia_FACTORS_c14,15)),min(tsrank(uq_liangjia_FACTORS_ACD6,25),delta(gaopin_liangjia_FACTORS_c125,25)))', 'rank(sign(ind_neutralize(gaopin_liangjia_FACTORS_c105)))', 'rank(sign(lowday(gaopin_liangjia_FACTORS_c79,30)))', 'ind_rank(delay(sign(choice_FACTORS_HQFW_SIGNAL),20))', '(kurtosis(median(jq_liangjia_FACTORS_boll_down,30),30)+((second_liangjia_FACTORS_bear+minute_FACTORS_weipan)*(gaopin_liangjia_FACTORS_c22-jq_liangjia_FACTORS_boll_down)))', 'ind_rank(sign(tsrank(uq_liangjia_FACTORS_ACD6,20)))', 'ind_rank(std(sign(jq_liangjia_FACTORS_boll_down),20))', 'tsrank(max(min(gaopin_liangjia_FACTORS_c13,gaopin_liangjia_FACTORS_c14),sign(second_liangjia_FACTORS_lag)),30)', 'ind_rank(sign(tsrank(uq_liangjia_FACTORS_MTM,20)))', 'kurtosis(delay(tsrank(fund_FACTORS_num,30),20),15)', 'rank(sign(lowday(gaopin_liangjia_FACTORS_c77,15)))']

    print(len(new_features))
    results = []
    ex = ProcessPoolExecutor(40)
    for tree in new_features:
        results.append(ex.submit(calculation, tree))
    ex.shutdown(wait=True)
    count = 0
    for res in results:
        tree = res.result()
        print(count, tree_to_formula(tree))
        tree.data.to_csv('/workspace1/liufengyuan/pct1_factors/data/alpha_%d.csv' % count)
        count += 1




import lightgbm as lgb
assert lgb.__version__ == '3.3.2'
import numpy as np
import glob
import pandas as pd
from joblib import Parallel, delayed
import time
import datetime
print(lgb.__version__)

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

def get_data():
    # 这个是做数据的columns的顺序，需要和训练时对齐
    start = time.time()
    a = np.load('/workspace1/liufengyuan/pct1_factors/feature_order.npy')
    alpha_list = []
    for num in range(0, 66):
        alpha_list.append('alpha_{}'.format(num))
    filenames = a.tolist()[:-66] + alpha_list
    filenames = list(map(lambda x: 'data/' + x + '.csv', filenames))
    result_list = []
    for file in filenames:
        name = file[5:-4]
        #print(name)
        if name == 'x_tech_liangjia_FACTORS_MF_20':
            result_list[-1][name] = np.nan
        elif name == 'minute_new_FACTORS_CDPDP':
            result_list[-1][name] = np.nan
        elif name == 'suntime_FACTORS_total_score':
            result_list[-1][name] = np.nan
        else:
            alpha = pd.read_csv(file)
            if alpha.columns.values.tolist()[0] == 'tradeDate':
                alpha = alpha.rename(columns={"tradeDate": "date"})
            alpha = alpha.set_index(['date']).stack().reset_index().rename(columns={'level_1': 'ticker', 0: name})
            alpha = alpha.sort_values(by=['ticker', 'date']).reset_index(drop=True)
            alpha = alpha.set_index(['ticker', 'date'])
            result_list.append(alpha)
    result = pd.concat(result_list, axis=1)
    print(f'耗时:{time.time() - start}')
    print('======== COMPLETED ========')
    result = reduce_mem_usage(result)
    print(result)
    return result

if __name__ == '__main__':
    import pickle
    save_path = '/workspace1/liufengyuan/pct1_factors/models/'
    models = []
    for i in range(10):
        file_name = save_path + 'gbm_model_%d.pickle' % i
        with open(file_name, 'rb') as file:
            gbm = pickle.load(file)
            models.append(gbm)
    print(models )
    data = get_data() #pd.read_csv('predict_result.csv')
    pred = np.zeros(data.shape[0])
    print(data)
    for gbm in models:
        pred += gbm.predict(data.values) / len(models)




    result = pd.DataFrame(index=data.index)
    result['prediction'] = pred
    today = (datetime.datetime.now()).strftime("%Y-%m-%d")
    result = result.reset_index()
    last_date = list(result.date.drop_duplicates())[-1]
    p_date = list(pd.to_datetime(result.date).drop_duplicates())[-1].strftime("%Y%m%d")
    result = result[result.date ==last_date ]
    result['date'] = p_date
    result.to_csv('LGBMRegressor_predict_value'+p_date+'.csv')
    result.to_csv('/workspace1/liufengyuan/pct1_factors/prediction_results/'+p_date+'.csv')
    print(result)


    #pickle.dump(pred, open("pred", "wb"))