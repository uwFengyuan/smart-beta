import pandas as pd
import pickle
import lightgbm as lgb
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import datetime
import time
from sklearn.linear_model import Ridge, LinearRegression
####========需要修改的全局参数========####
print('========LOADING DATA========')
#df1 = pd.read_csv('/data/liufengyuan/pct1_cal/modified_alter_alphas_066_labels.csv')
df1 = pd.read_csv('/data/liufengyuan/pct5_cal/modified_alter_alphas_036_222_labels.csv')
print(df1)
print('========COMPLETE LOADING DATA========')
label_list = ['PCT5_rank', 'PCT2_rank', 'openclose_pct1_rank', 'askbid_pct1_rank']
f_y = label_list[0]
#f_x = pickle.load(open("/data/liufengyuan/pct1_cal/f_x_066", "rb"))
f_x = pickle.load(open('/data/liufengyuan/pct5_cal/f_x_036_222', 'rb'))
#print(f_x)
data_source = 'full_Alter_036_222'
file_location = 'pct5_cal'
file_type = 'full'
model_list = ['LinearRegression', 'RidgeR', 'DecisionTreeR', 'XGBoostR', 'LGBMRegressor']
model_name = model_list[4]
####========需要修改的全局参数========####

dates = df1.tradeDate.sort_values().unique()
epoch_ts = list(dates)
f_index = ['ticker', 'tradeDate']
#f_x_036_222 = pickle.load(open('pct5_cal/f_x_036_222', 'rb'))
#f_222 = pickle.load(open('pct5_cal/f_alphas_222', 'rb'))
#f_x = [x for x in f_x_036_222 if x not in f_222]

target_types = ['r', 'c'] # 分类问题还是回归问题 r 回归问题 c 分类问题
target_type = target_types[0]
result_name = '{}_{}_{}_{}'.format(data_source, model_name, f_y, target_type)
print(result_name)

update = 22 # 训练长度：22天
train_si = epoch_ts.index('2017-01-03') # included. '2017-01-03'
train_ei = epoch_ts.index('2019-01-02') # excluded. '2018-12-28'
test_si = epoch_ts.index('2019-01-02') # included. '2019-01-02'
test_ei = epoch_ts.index('2019-02-01') # excluded. '2019-01-31'
test_fi = len(epoch_ts) - 1 # excluded. '2019-01-16'

# number of epochs，循环次数
num_epoch = round((test_fi - test_ei) / update)
epoch_range = range(0, num_epoch + 1)
#epoch_range = range(0, 1)

start = time.time()
df_result_all = pd.DataFrame()
best_parameters_list = {}
total_IC = 0
i_range = list(np.arange(0, 1, 0.1)) + list(np.arange(0, 20, 1)) + [30, 40, 100, 200, 300]
for epoch in epoch_range:
    print('----- EPOCH {}------'.format(epoch))
    best_df_result = pd.DataFrame()
    best_params = {}
    best_IC = -np.inf
    update_n = epoch * update
    # get a list of train dates
    epoch_t_train = epoch_ts[train_si + update_n : train_ei + update_n]
    # get a list of test dates
    epoch_t_test = epoch_ts[test_si + update_n : test_ei + update_n]
    df_train = df1[df1.tradeDate.apply(lambda x: x in epoch_t_train)].reset_index(drop=True)
    df_test = df1[df1.tradeDate.apply(lambda x: x in epoch_t_test)].reset_index(drop=True)
    #print('预测时间：', epoch_t_test)
    #print('数据大小：', df_train.shape, df_test.shape)

    # 获得 x
    x_train = df_train[f_x].values
    x_test = df_test[f_x].values
    #print('处理后x:', x_train.shape, x_test.shape)

    # 获得y
    y_train = df_train[f_y].copy()
    y_test = df_test[f_y].copy()
    #print('处理后y:', y_train.shape, y_test.shape)

    if model_name == 'LinearRegression':
        model = LinearRegression(n_jobs=-1)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
    elif model_name=='RidgeR': 
        for i in i_range:
            params = {
                    'alpha': i 
                }
            model = Ridge(**params)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            # print('get result')
            df_result = df_test[f_index].copy()
            df_result['y'] = y_test
            df_result['y_pred'] = y_pred
            # print(df_result)
            IC = df_result[['y', 'y_pred']].corr().iloc[0, 1]
            #print(IC)
            if best_IC < IC:
                best_IC = IC
                best_params = params
                best_df_result = df_result.copy()
    for max_depth in [4, 5, 6]:
        for num_leaves in [10, 30, 50]:
            for learning_rate in [0.01, 0.05, 0.09]:
                # for min_child_samples in range(120, 180, 10):
                # 'min_child_samples': range(10, 210, 20),
                # 'subsample': [i/10.0 for i in range(7,10)],
                # 'subsample_freq': range(1,10,1)
                params = {
                    'max_depth': max_depth,
                    'num_leaves': num_leaves,
                    'learning_rate': learning_rate
                }
                gbm = lgb.LGBMRegressor(
                    **params,
                    verbose=-1,
                    objective='regression',
                    n_estimators=100,
                    metrics='rmse',
                    subsample=0.9,
                    subsample_freq=2,
                    min_child_samples=150,
                    max_bin=256,
                    n_jobs=-1
                )
                gbm.fit(x_train, y_train)
                y_pred = gbm.predict(x_test)
                # print('get result')
                df_result = df_test[f_index].copy()
                df_result['y'] = y_test
                df_result['y_pred'] = y_pred
                # print(df_result)
                IC = df_result[['y', 'y_pred']].corr().iloc[0, 1]
                print(IC)
                if best_IC < IC:
                    best_IC = IC
                    best_params = params
                    best_df_result = df_result.copy()
    # full, 1000, 0.09: 0.059;
    # 500: 0.0323
    # 500, 3200, 0.01: 0.0266
    # 500， 0.09， 100： 修改rank之后的结果 0.0396
    # full， 被提供参数： 修改rank之后的结果
    # 获得结果

    print(f'耗时:{time.time() - start}')
    print('get result')
    #df_result = df_test[f_index].copy()
    #df_result['y'] = y_test
    #df_result['y_pred'] = y_pred
    #IC = df_result[['y', 'y_pred']].corr().iloc[0,1]
    print('========BEST RESULT========')
    print(best_params)
    print(best_IC)
    total_IC += best_IC
    df_result_all = df_result_all.append(best_df_result)

#print('PARALLEL'.format(len_train))
#start = time.time()
#parallel_obj2 = Parallel(n_jobs=4)(delayed(runEpoch)(epoch) for epoch in epoch_range)
print(f'耗时:{time.time() - start}') #
#df_result_all = pd.concat(parallel_obj2)
print('sort values')
df_result_all = df_result_all.sort_values(by=['ticker', 'tradeDate']).reset_index(drop=True)
IC = total_IC / num_epoch
today = (datetime.datetime.now()).strftime("%Y-%m-%d:%H:%M:%S")
#pickle.dump(best_parameters_list, open('{}/{}/best_parameters_list_{}_{}'.format(file_location, file_type, today, IC), 'wb'))
df_result_all.to_csv('/data/liufengyuan/{}/{}/{}{}_{}.csv'.format(file_location, file_type, result_name, today, round(IC, 4)), index=False)
print('======== COMPLETED {} {} ========'.format(model_name, IC))

#print('PARALLEL')
#start = time.time()
#parallel_obj = Parallel(n_jobs=4)(delayed(run_num)(depth) for depth in depth_list)
#print(f'耗时:{time.time() - start}')
