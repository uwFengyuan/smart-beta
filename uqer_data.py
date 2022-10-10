import  numpy as np
import pandas as pd
import uqer
import pickle
from joblib import Parallel, delayed
import time
import glob
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from uqer import DataAPI   #优矿api
import lightgbm as lgb
client = uqer.Client(token='18266a7c0ac9f8cdbe00f9b2ecb65f42316a5f78d9cc22ebabcbd923593356e4')

# a list of functions
## convert ticker numbers into strings
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
    factors = DataAPI.MktStockFactorsDateRangeProGet(ticker=ticker_name, beginDate=u"20170103",
                                                     endDate=u"20220405", pandas="1")
    return factors

print('load data')
data = pd.read_csv('4uqer_idst_log.csv')
print(data)
strData = ticekrToStr(data)

def getData(tempticker):
    print(tempticker)
    return get_target(tempticker)

print('PARALLEL')
start = time.time()
parallel_obj = Parallel(n_jobs=48)(delayed(getData)(tempticker) for tempticker in strData.ticker.unique())
print(f'耗时:{time.time() - start}')
df_merge = pd.concat(parallel_obj)
df_merge = df_merge.iloc[:, 1:]
print('sort values')

#df_merge = df_merge.merge(data, on=['ticker', 'tradeDate'], how='left')
#df_merge = df_merge.sort_values(by=['ticker', 'tradeDate']).reset_index(drop=True)
print(df_merge)
df_merge = reduce_mem_usage(df_merge)
df_merge.to_csv('uqer_cal/uqer_data.csv', index = False)

print('load data')
data = pd.read_csv('4uqer_idst_log.csv')
df_merge = pd.read_csv('uqer_cal/uqer_data.csv')

df_merge = df_merge.merge(data, on=['ticker', 'tradeDate'], how='left')
df_merge = df_merge.sort_values(by=['ticker', 'tradeDate']).reset_index(drop=True)
df_merge = reduce_mem_usage(df_merge)
df_merge.to_csv('uqer_cal/uqer_idst_log_data.csv', index = False)

print('get column lists')
f_index = ['ticker', 'tradeDate']
f_industry_uqer = pickle.load(open("f_industry", "rb"))
pickle.dump(f_industry_uqer, open("uqer_cal/f_industry_uqer", "wb"))
f_x_uqer = df_merge.columns.values.tolist()[2:]
print(f_x_uqer)
pickle.dump(f_x_uqer, open("uqer_cal/f_x_uqer", "wb"))
label_list = ['PCT5_rank', 'PCT2_rank', 'openclose_pct1_rank', 'askbid_pct1_rank']

print('========COMBINING LABEL DATA========')
path2 = 'labels'
filenames2 = glob.glob(path2 + '/*.csv')
for file in filenames2:
    print(file)
    label = pd.read_csv(file)
    label_name = label.columns.values.tolist()[-1]
    label = label[['ticker', 'tradeDate', label_name]]
    df_merge = df_merge.merge(label, on=['ticker', 'tradeDate'], how='left')
    del label
print(df_merge.shape)
print('========COMPLETE COMBINING LABEL DATA========')
#df_merge = reduce_mem_usage(df_merge)
alter = reduce_mem_usage(df_merge)
del df_merge
print('raw store data')
#df_merge.to_csv('uqer_cal/uqer_labels_raw.csv', index = False)
alter.to_csv('uqer_cal/uqer_labels_raw.csv', index = False)

####========####
####========####
####========####
print('========LOADING DATA========')
alter = pd.read_csv('uqer_cal/uqer_labels_raw.csv')
print(alter)
print('========COMPLETE LOADING DATA========')

print('get column lists')
f_index = ['ticker', 'tradeDate']
f_industry = pickle.load(open("uqer_cal/f_industry_uqer", "rb"))
f_x = pickle.load(open("uqer_cal/f_x_uqer", "rb"))
label_list = ['PCT5_rank', 'PCT2_rank', 'openclose_pct1_rank', 'askbid_pct1_rank']

# %% a list of classes
## exclude extreme values
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

## fill empty values
class FillEmpty(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.df_mean_industry = None

    def fit(self, df_name, f_x = f_x, group_name='industry'):
        self.df_mean_industry = df_name.groupby(group_name).mean()[f_x]
        self.df_mean_industry = self.df_mean_industry.fillna(self.df_mean_industry.mean())
        self.df_mean_industry.columns = [x + '_mean' for x in self.df_mean_industry.columns]
        self.df_mean_industry = self.df_mean_industry.reset_index()
        self.df_mean_industry = self.df_mean_industry.fillna(0)  # 可以删除
        return self

    def transform(self, df_name, f_x = f_x, group_name='industry'):
        df_name_mean = df_name.merge(self.df_mean_industry, on=group_name, how='left')
        df_name_mean[f_x] = df_name_mean[f_x].apply(lambda x: x.fillna(df_name_mean[x.name + '_mean']))
        df_name = df_name_mean[df_name.columns]
        return df_name

## neutralize industry
class Neutralization(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.models = {}

    def regr_fit(self, X, y):
        self.models[y.name] = linear_model.LinearRegression().fit(X, y)
        return

    def regr_pred(self, X, y):
        pred = self.models[y.name].predict(X)
        return y - pred

    def fit(self, df_name, f_x = f_x, f_idst = f_industry):
        X = df_name[['log_marketValue'] + f_idst]
        df_name[f_x].apply(lambda y: self.regr_fit(X, y))
        return self

    def transform(self, df_name, f_x = f_x, f_idst = f_industry):
        X = df_name[['log_marketValue'] + f_idst]
        df_name[f_x] = df_name[f_x].apply(lambda y: self.regr_pred(X, y))
        return df_name

dates = alter.tradeDate.sort_values().unique()
timediff = pd.Timedelta(100,unit='d')

def run_df(date):
#for date in dates:
    print('----- DATE {}------'.format(date))
    df2 = alter[alter.tradeDate == date]
    df2 = df2[~df2.log_marketValue.isnull()]
    print('处理前无效值：', df2[label_list].isnull().any().sum())
    df2 = df2[~df2.PCT2_rank.isnull()]
    df2 = df2[~df2.PCT5_rank.isnull()]
    df2 = df2[~df2.askbid_pct1_rank.isnull()]
    df2 = df2[~df2.openclose_pct1_rank.isnull()]
    print('处理后无效值：', df2[label_list].isnull().any().sum())

    # 数据筛选 删除上市100天以内的
    a = pd.to_datetime(df2.tradeDate)
    b = pd.to_datetime(df2.listData)
    df2 = df2[a-b > timediff]

    # 中位数去极值
    print('exclude extreme values')
    mevtransformer = ExcludeExtreme()
    mevtransformer.fit(df2[f_x])
    df2[f_x] = mevtransformer.transform(df2[f_x])

    # 缺失值处理
    print('deal with null values')
    gvfiller = FillEmpty()
    gvfiller = gvfiller.fit(df2)
    df2 = gvfiller.transform(df2)
    print('缺失值：', df2[f_x].isnull().any().sum())

    # 行业中性处理
    print('do neutralization')
    idst_neutral = Neutralization()
    idst_neutral = idst_neutral.fit(df2)
    df2 = idst_neutral.transform(df2)

    # 标准化处理
    print('do standardization')
    scaler = StandardScaler()
    scaler.fit(df2[f_x])
    df2[f_x] = scaler.transform(df2[f_x])
    df2 = reduce_mem_usage(df2)
    df2 = df2[f_index + f_x + label_list]
    return df2
    #df2.to_csv('train_test/{}_data.csv'.format(date), index = False)

print('PARALLEL')
start = time.time()
parallel_obj = Parallel(n_jobs=8)(delayed(run_df)(date) for date in dates)
print(f'耗时:{time.time() - start}')
del alter
df1 = pd.concat(parallel_obj)
print('sort values')
df1 = df1.sort_values(by=['ticker', 'tradeDate']).reset_index(drop=True)
print('reduce memory')
df1 = reduce_mem_usage(df1)
print('store data')
# df1.to_csv('train_test/modified_alter_alphas_036_222_labels.csv', index = False)
df1.to_csv('uqer_cal/modified_uqer_labels.csv', index = False)

####========####
####========####
####========####
print('========LOADING DATA========')
# data = pd.read_csv('train_test/modified_alter_alphas_036_222_labels.csv', chunksize=500000)
print(df1)
print('========COMPLETE LOADING DATA========')

dates = df1.tradeDate.sort_values().unique()
print('get epoch list')
epoch_ts = list(dates)

# ------- 需要测试的全局参数 ------ #
if_pcas = ['', 'pca'] # pca 或者空字符串# 是否做PCA
if_pca = if_pcas[1] # pca 或者空字符串
pca_components_list = [0.99, 0.95, 0.90, 0.85, 0.80]
pca_components = pca_components_list[0]
num_leaves_list = [10, 15, 20, 25, 30]
num_leaves = num_leaves_list[0]
for f_y in label_list:
    print('======== LEN_TRAIN {} ========'.format(f_y))
    target_types = ['r', 'c'] # 分类问题还是回归问题 r 回归问题 c 分类问题
    target_type = target_types[0]
    model_name = 'LGBMRegressor'

    result_name = 'UQER_{}-{}-{}-{}-{}-{}'.format(model_name, f_y, target_type, num_leaves, if_pca, int(100*pca_components))
    print(result_name)

    update = 22 # 训练长度：22天
    train_si = epoch_ts.index('2017-01-03') # included. '2017-01-03'
    train_ei = epoch_ts.index('2019-01-02') # excluded. '2018-12-28'
    test_si = epoch_ts.index('2019-01-02') # included. '2019-01-02'
    test_ei = epoch_ts.index('2019-02-01') # excluded. '2019-01-31'
    test_fi = len(epoch_ts) - 1 # excluded.

    # number of epochs，循环次数
    num_epoch = round((test_fi - test_ei) / 22)
    epoch_range = range(0, num_epoch + 1)

    start = time.time()
    df_result_all = pd.DataFrame()
    for epoch in epoch_range:
        print('----- EPOCH {}------'.format(epoch))
        update_n = epoch * update
        # get a list of train dates
        epoch_t_train = epoch_ts[train_si + update_n : train_ei + update_n]
        # get a list of test dates
        epoch_t_test = epoch_ts[test_si + update_n : test_ei + update_n]
        df_train = df1[df1.tradeDate.apply(lambda x: x in epoch_t_train)].reset_index(drop=True)
        df_test = df1[df1.tradeDate.apply(lambda x: x in epoch_t_test)].reset_index(drop=True)
        print('预测时间：', epoch_t_test)
        print('数据大小：', df_train.shape, df_test.shape)

        # 数据筛选 删除target为缺失值的
        print('缺失值：', df_train.isnull().any().sum())
        print('缺失值：', df_test.isnull().any().sum())
        #print('delete rows without target')
        #df_train = df_train[~df_train.target.isnull()]
        #df_test = df_test[~df_test.target.isnull()]

        # 获得 x
        # PCA处理
        if if_pca == 'pca':
            from sklearn.decomposition import PCA

            pca = PCA(n_components=pca_components)
            pca.fit(df_train[f_x])
            x_train = pca.transform(df_train[f_x])
            x_test = pca.transform(df_test[f_x])
        else:
            x_train = df_train[f_x].values
            x_test = df_test[f_x].values
        print('处理后x：', x_train.shape, x_test.shape)

        # 获得y
        y_train = df_train[f_y].copy()
        y_test = df_test[f_y].copy()
        print('处理后y：', y_train.shape, y_test.shape)

        model = lgb.LGBMRegressor(learning_rate=0.09, num_leaves = num_leaves, max_depth=5)
        model.fit(x_train, y_train, eval_set=[(x_train,y_train), (x_test,y_test)], eval_metric='l2')
        y_pred = model.predict(x_test)

        # 获得结果
        print('get result')
        df_result = df_test[f_index].copy()
        df_result['y'] = y_test
        df_result['y_pred'] = y_pred
        #df_result.to_csv('5_result{}.csv'.format(epoch), index=False)
        #return df_result
        df_result_all = df_result_all.append(df_result)

    #print('PARALLEL'.format(len_train))
    #start = time.time()
    #parallel_obj2 = Parallel(n_jobs=4)(delayed(runEpoch)(epoch) for epoch in epoch_range)
    print(f'耗时:{time.time() - start}') # 1858.926 秒 [5241.98, ]
    #df_result_all = pd.concat(parallel_obj2)
    print('sort values')
    df_result_all = df_result_all.sort_values(by=['ticker', 'tradeDate']).reset_index(drop=True)
    df_result_all = reduce_mem_usage(df_result_all)
    print('store data')
    df_result_all.to_csv('results_all/{}.csv'.format(result_name), index=False)
    print('======== COMPLETED {} ========'.format(f_y))