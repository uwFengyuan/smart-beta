import time
import pandas as pd
import pickle
import glob
import numpy as np
from functools import reduce
from joblib import Parallel, delayed

alter = pd.read_csv('pct1_cal/1alter_alphas_066.csv')
print('========LOADING UQER DATA========')
uqer_data = pd.read_csv('pct1_cal/4uqer_idst_log.csv')
print(uqer_data.shape)
print('========COMPLETE LOADING UQER DATA========')

print('========COMBINE ALTER ALPHA UQER DATA========')
alter = alter.merge(uqer_data, on=['ticker', 'tradeDate'], how='left')
alter = alter[~alter.log_marketValue.isnull()]
print(alter.shape)
del uqer_data
print('========COMPLETE COMBINE ALTER ALPHA UQER DATA========')

print('get column lists')
f_index = ['ticker', 'tradeDate']
f_industry = pickle.load(open("pct1_cal/f_industry", "rb"))
f_x = pickle.load(open("pct1_cal/f_x_066", "rb"))
label_list = ['askbid_pct1_rank', 'openclose_pct1_rank']

print('========COMBINING LABEL DATA========')
label = pd.read_csv('labels/openclose_pct1.csv')
label_name = label.columns.values.tolist()[-1]
label = label[['ticker', 'tradeDate', label_name]]
alter = alter.merge(label, on=['ticker', 'tradeDate'], how='left')
print(alter.shape)
print('========COMPLETE COMBINING LABEL DATA========')

print('store raw data')
alter.to_csv('pct1_cal/alter_idst_alphas_066_labels_raw.csv', index = False)

"""
print('========LOADING ALTER DATA========')
alter = pd.read_csv('alternative_factors.csv')
alter = alter.iloc[:, 1:]
alter = alter.dropna(thresh = 67)
print(alter.shape)
print('========COMPLETE LOADING ALTER DATA========')

print('========LOADING UQER DATA========')
uqer_data = pd.read_csv('4uqer_idst_log.csv')
print(uqer_data.shape)
print('========COMPLETE LOADING UQER DATA========')

print('========COMBINE ALTER UQER DATA========')
alter = alter.merge(uqer_data, on=['ticker', 'tradeDate'], how='left')
alter = alter[~alter.log_marketValue.isnull()]
print(alter.shape)
del uqer_data
alter = reduce_mem_usage(alter)
print('========COMPLETE COMBINE ALTER UQER DATA========')

print('========COMBINING ALPHA DATA========')
filenames = glob.glob('temp_combine_alphas_*')
print(filenames)
print('LOOP:')
for file in filenames:
    print(file)
    alpha = pd.read_csv(file)
    alpha = alpha.sort_values(by=['ticker', 'tradeDate']).reset_index(drop=True)
    alter = alter.merge(alpha, on=['ticker', 'tradeDate'], how='left')
    del alpha
print(alter.shape)
alter = reduce_mem_usage(alter)
print('========COMPLETE COMBINING ALPHA DATA========')

print('get column lists')
f_index = ['ticker', 'tradeDate']
f_industry = pickle.load(open("f_industry", "rb"))
f_x = pickle.load(open("f_x", "rb"))
label_list = ['PCT5_rank', 'PCT2_rank', 'openclose_pct1_rank', 'askbid_pct1_rank']

print('========COMBINING LABEL DATA========')
path2 = 'labels'
filenames2 = glob.glob(path2 + '/*.csv')
for file in filenames2:
    print(file)
    label = pd.read_csv(file)
    label_name = label.columns.values.tolist()[-1]
    label = label[['ticker', 'tradeDate', label_name]]
    alter = alter.merge(label, on=['ticker', 'tradeDate'], how='left')
    del label
print(alter.shape)
print('========COMPLETE COMBINING LABEL DATA========')
alter = reduce_mem_usage(alter)
print('raw store data')
alter.to_csv('alter_idst_alphas_036_222_labels_raw.csv', index = False) #(4778142 ,522)
"""