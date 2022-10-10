import glob
import pandas as pd
from functools import reduce
from joblib import Parallel, delayed
import time
import numpy as np
import pickle
import uqer
from uqer import DataAPI   #优矿api
import lightgbm as lgb
#assert lgb.__version__ == '3.3.2'
import sklearn
#client = uqer.Client(token='18266a7c0ac9f8cdbe00f9b2ecb65f42316a5f78d9cc22ebabcbd923593356e4')
"""
df_train = pd.read_csv('pct1_cal/modified_alter_alphas_066_labels.csv', nrows=1).columns.values.tolist()
print(df_train)
quit()

print('get the names of columns')
f_index = ['ticker', 'tradeDate']
f_x = pickle.load(open("f_x", "rb"))
print(f_x)
print(len(f_x))
col_list = f_index + f_x
print('load data')
df = pd.read_csv('6filled_modified_data.csv', usecols=col_list)
print('store data')
df.to_csv('train_test/modified_alter_alphas_036.csv', index=False)
print('========COMPLETE STORING========')
quit()
path = 'alphas_all'
filenames = glob.glob(path + '/*.csv')
factors = 0
def run(file):
    print(file)
    alpha = pd.read_csv(file)
    name = file[11:-4]
    empty = alpha[name].isnull().sum()
    total = alpha[name].shape[0]
    ratio = empty / total
    return ratio

print('PARALLEL')
start = time.time()
ratio_list = Parallel(n_jobs=40)(delayed(run)(file) for file in filenames)
print(f'耗时:{time.time() - start}')
print('Minimum ratio: {}'.format(min(ratio_list)))
print('Maximum ratio: {}'.format(max(ratio_list)))
print('Total number of valid features: {}'.format(len([x for x in ratio_list if x < 0.3])))
quit()
file = 'alphas_all/222_alpha_11.csv'
alpha = pd.read_csv(file)
alpha = alpha.sort_values(by=['ticker', 'tradeDate']).reset_index(drop=True)
print(alpha)
name = file[11:-4]
print(name)
print(alpha[name].isnull().sum())
print(alpha[name].shape[0])
quit()
"""
"""
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
data = pd.read_csv('labels/askbid_pct1.csv')
print(data)
data = pd.read_csv('labels/openclose_pct1.csv')
print(data)
data = pd.read_csv('results_all/Alter_066_LGBMRegressor-askbid_pct1_rank-r-10--99.csv')
data = data[['ticker', 'tradeDate', 'y']]
print(data)
data = pd.read_csv('results_all/Alter_066_LGBMRegressor-openclose_pct1_rank-r-10--99.csv')
data = data[['ticker', 'tradeDate', 'y']]
print(data)
data1 = pd.read_csv('pct1_cal/Alter_066_LGBMRegressor-openclose_pct1_rank-r-10--99.csv')
data1 = data1[['ticker', 'tradeDate', 'y']]
print(data1)
out = list(set(data1['ticker']) - set(data['ticker']))
print(out)
print(len(out))
data2 = pd.read_csv('pct1_cal/1alter_alphas_066.csv')
for element in out:
    print(data2[data2.ticker == element])

quit()
a = np.load('./feature_order.npy')
alpha_list = []
for num in range(0, 66):
    alpha_list.append('alpha_{}'.format(num))
factors = a.tolist()[:-66] + alpha_list
#factors = list(map(lambda x: x.replace('suntime_FACTORS_total_score', 'suntime_FACTORS_score'), factors))
#factors = list(map(lambda x: x.replace('minute_new_FACTORS_CDPDP', 'minute_new_FACTORS_CDPP'), factors))
#factors = list(map(lambda x: x.replace('x_tech_liangjia_FACTORS_MF_20', 'x_tech_liangjia_FACTORS_MF20'), factors))
#filenames = list(map(lambda x: 'temp_data_set/' + x + '.csv', filenames))

path = 'data'
list_data = glob.glob(path + '/*.csv')
names = []
for file in list_data:
    name = file[5:-4]
    names.append(name)
print(len(names))
count = 0
for file in factors:
    if file not in names:
        print(file)
        count+=1
print(count)

count = 0
for file in names:
    if file not in factors:
        print(file)
        count+=1
print(count)

quit()


print("Sklearn verion is {}".format(lgb.__version__))
quit()
temp = pd.read_csv('predict_result2.csv')
print(temp)
temp = pd.read_csv('predict_result3.csv')
print(temp)
temp = pd.read_csv('predict_result_stack_concat.csv')
temp = temp.sort_values(by=['ticker', 'date']).reset_index(drop=True)
print(temp)
quit()

#outData = DataAPI.MktStockFactorsDateRangeGet(ticker=u"000001",beginDate=u"20170612",endDate=u"20170616",field=u"",pandas="1")
#filenames = glob.glob('results_all/*99.csv')
#for file in filenames:
#    name = file[12:-21]
    #if name == "UQER_LGBMRegressor-PCT5":
    #    name = "UQER_LGBMRegressor-PCT5_222_036"
    #name = name.replace('LGBMRegressor-', 'Prediction_')
#    print('load {} data'.format(name))
"""

factors = pickle.load(open("pct1_cal/f_x_066", "rb")) #alternative

path = 'data' #data
list_data = glob.glob(path + '/*.csv')
names = []
for file in list_data:
    name = file[5:-4]
    names.append(name)
print(len(names))

a = np.load('./feature_order.npy')
alpha_list = []
for num in range(0, 66):
    alpha_list.append('alpha_{}'.format(num))
factors2 = a.tolist()[:-66] + alpha_list

count = 0
print('In alternative not in feature_order: ')
for file in factors:
    if file not in factors2:
        print(file)
        count+=1
print(count)

count = 0
print('In feature_order not in alternative: ')
for file in factors2:
    if file not in factors:
        print(file)
        count+=1
print(count)

count = 0
print('In alternative not in data: ')
for file in factors:
    if file not in names:
        print(file)
        count+=1
print(count)

count = 0
print('In data not in alternative: ')
for file in names:
    if file not in factors:
        print(file)
        count+=1
print(count)

quit()

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

import uqer
from uqer import DataAPI   #优矿api
client = uqer.Client(token='18266a7c0ac9f8cdbe00f9b2ecb65f42316a5f78d9cc22ebabcbd923593356e4')
all_data = DataAPI.IdxConsCoreGet(secID=u"",ticker=u"000905",intoDate=u"",outDate=u"",field=u"",pandas="1")

all_data['consTickerSymbol'] = all_data['consTickerSymbol'].astype(int)
tickers_500 = all_data.consTickerSymbol.sort_values().unique()
print(tickers_500.shape)

df1 = pd.read_csv('pct1_cal/modified_alter_alphas_066_labels_500.csv')
print(df1)

print('load openclose')
openclose = pd.read_csv('labels/openclose_pct1.csv')
print(openclose)
openclose = openclose [openclose.ticker.apply(lambda x: x in tickers_500)]

openclose['openclose_pct1_rank'] = openclose.groupby('tradeDate')['openclose_pct1'].rank(pct = True)
openclose = openclose.drop(columns='openclose_pct1')

df1 = df1.drop(columns='openclose_pct1_rank')
df1 = df1.merge(openclose, on=['ticker', 'tradeDate'], how='left')
print(df1)
df1 = reduce_mem_usage(df1)
df1.to_csv('pct1_cal/modified_alter_alphas_066_labels_5002.csv', index = False)
quit()


filenames = glob.glob('pct1_cal/500*.csv')
print(filenames)
def run(file):
#for file in filenames:
    out = pd.read_csv(file)
    name = file[9:-8]
    print('load {} data'.format(name))
    dates = out.tradeDate.sort_values().unique()
    total = 0
    for date in dates:
        temp_result = out[out.tradeDate == date]
        # 加入到合并数据
        IC = temp_result[['y', 'y_pred']].corr().iloc[0,1]
        total += IC
        # print('----- EPOCH {}: IC {}------'.format(date, IC))
    print('----- {} Average IC {}------'.format(name, total/(len(list(dates)))))

print('PARALLEL')
start = time.time()
parallel_obj = Parallel(n_jobs=10)(delayed(run)(file) for file in filenames)
print(f'耗时:{time.time() - start}')
print('----- COMPLETED------')
    # PCT5_rank: 0.1241881983956163
"""
load LGBMRegressor-PCT5 data
----- Average IC 0.1241881983956163------
load LGBMRegressor-askbid_pct1 data
----- Average IC 0.23151250652410377------
load UQER_LGBMRegressor-PCT5 data
----- Average IC 0.1050115846887966------
load LGBMRegressor-PCT2 data
----- Average IC 0.10322197725116287------
load UQER_LGBMRegressor-openclose_pct1 data
----- Average IC 0.040404361072185736------
load UQER_LGBMRegressor-PCT2 data
----- Average IC 0.0889521135000151------
load UQER_LGBMRegressor-askbid_pct1 data
----- Average IC 0.21250969231795774------
load LGBMRegressor-openclose_pct1 data
----- Average IC 0.05060571450101373------
----- COMPLETED------
中证500，用500，uqer，指数成分股。每天的成分股，merge
"""

"""
Alter+Alpha_066
----- Alter_066_LGBMRegressor-askbid_pct1_rank-r-10 Average IC 0.26001677573656357------
----- Alter_066_LGBMRegressor-askbid_pct1_rank-r-15 Average IC 0.2613215568711985------
----- Alter_066_LGBMRegressor-askbid_pct1_rank-r-20 Average IC 0.2618731831394444------
----- Alter_066_LGBMRegressor-askbid_pct1_rank-r-25 Average IC 0.2621057342411864------
----- Alter_066_LGBMRegressor-askbid_pct1_rank-r-30 Average IC 0.26203405679250386------
----- Alter_066_LGBMRegressor-openclose_pct1_rank-r-10 Average IC 0.06083189588982539------
----- Alter_066_LGBMRegressor-openclose_pct1_rank-r-15 Average IC 0.0625877462413099------
----- Alter_066_LGBMRegressor-openclose_pct1_rank-r-20 Average IC 0.062192001178555216------
----- Alter_066_LGBMRegressor-openclose_pct1_rank-r-25 Average IC 0.06290215854084702------
----- Alter_066_LGBMRegressor-openclose_pct1_rank-r-30 Average IC 0.06223776199270528------
"""
