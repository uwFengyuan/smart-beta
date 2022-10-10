import time
import pandas as pd
import pickle
import numpy as np
import glob
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

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
import uqer
from uqer import DataAPI   #优矿api
client = uqer.Client(token='18266a7c0ac9f8cdbe00f9b2ecb65f42316a5f78d9cc22ebabcbd923593356e4')
all_data = DataAPI.IdxConsGet(secID=u"",ticker=u"000852",isNew=u"",intoDate=u"",field=u"",pandas="1")
all_data['consTickerSymbol'] = all_data['consTickerSymbol'].astype(int)
tickers_1000 = all_data.consTickerSymbol.sort_values().unique()
print(tickers_1000.shape)

print('load openclose')
openclose = pd.read_csv('labels/openclose_pct1.csv')
print(openclose)
openclose = openclose [openclose.ticker.apply(lambda x: x in tickers_1000)]
openclose['openclose_pct1_rank'] = openclose.groupby('tradeDate')['openclose_pct1'].rank(pct = True)
openclose = openclose.drop(columns='openclose_pct1')
print(openclose)

df_1000 = pd.read_csv('pct1_cal/alter_idst_alphas_066_labels_raw_1000.csv')
df_1000 = df_1000.drop(columns='openclose_pct1_rank')

df_1000 = df_1000.merge(openclose, on=['ticker', 'tradeDate'], how='left')
print(df_1000)
df_1000.to_csv('pct1_cal/alter_idst_alphas_066_labels_raw_1000.csv')
"""
df = pd.read_csv('pct1_cal/alter_idst_alphas_066_labels_raw2.csv')
print(df)

print('get column lists')
f_index = ['ticker', 'tradeDate']
f_industry = pickle.load(open("pct1_cal/f_industry", "rb"))
f_x = pickle.load(open("pct1_cal/f_x_066", "rb"))
label_list = ['openclose_pct1'] #['PCT5_rank', 'PCT2_rank', 'openclose_pct1_rank', 'askbid_pct1_rank']

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

dates = df.tradeDate.sort_values().unique()
timediff = pd.Timedelta(100,unit='d')

def run_df(date):
#for date in dates:
    print('----- DATE {}------'.format(date))
    df2 = df[df.tradeDate == date]
    print('处理前无效值：', df2[label_list].isnull().any().sum())
    df2 = df2[~df2.log_marketValue.isnull()]
    #df2 = df2[~df2.PCT2_rank.isnull()]
    #df2 = df2[~df2.PCT5_rank.isnull()]
    #df2 = df2[~df2.askbid_pct1_rank.isnull()]
    df2 = df2[~df2.openclose_pct1.isnull()]
    print('处理后无效值：', df2[label_list].isnull().any().sum())

    # 数据筛选 删除上市100天以内的
    #a = pd.to_datetime(df2.tradeDate)
    #b = pd.to_datetime(df2.listData)
    #df2 = df2[a-b > timediff]

    # 中位数去极值
    print('exclude extreme values')
    mevtransformer = ExcludeExtreme()
    mevtransformer.fit(df2[f_x])
    df2[f_x] = mevtransformer.transform(df2[f_x])

    # 缺失值处理
    print('deal with null values')
    #gvfiller = FillEmpty()
    #gvfiller = gvfiller.fit(df2)
    #df2 = gvfiller.transform(df2)
    df2 = df2.fillna(0)
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

    df2 = df2[f_index + f_x + label_list + ['listData']]
    #df1 = df1.append(df2)
    return df2
    #df2.to_csv('train_test/{}_data.csv'.format(date), index = False)

print('PARALLEL')
start = time.time()
parallel_obj = Parallel(n_jobs=48)(delayed(run_df)(date) for date in dates)
print(f'耗时:{time.time() - start}')
df1 = pd.concat(parallel_obj)
print('sort values')
df1 = df1.sort_values(by=['ticker', 'tradeDate']).reset_index(drop=True)
print('reduce memory')
df1 = reduce_mem_usage(df1)
print('store data')
# df1.to_csv('train_test/modified_alter_alphas_036_222_labels.csv', index = False)
# df1.to_csv('uqer_cal/modified_uqer_labels.csv', index = False)
df1.to_csv('pct1_cal/modified_alter_alphas_066_labels2.csv', index = False)