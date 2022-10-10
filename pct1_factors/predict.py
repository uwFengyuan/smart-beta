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
    filenames = list(map(lambda x: 'pct1_factors/data/' + x + '.csv', filenames))
    result_list = []
    for file in filenames:
        name = file[18:-4]
        print(name)
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
    print(models)
    data = get_data() #pd.read_csv('predict_result.csv')
    pred = np.zeros(data.shape[0])
    print(data)
    for gbm in models:
        pred += gbm.predict(data.values) / len(models)




    result = pd.DataFrame(index=data.index)
    result['prediction'] = pred
    today = (datetime.datetime.now()).strftime("%Y-%m-%d")
    result = result.reset_index()
    result['ticker'] = result['ticker'].astype(int)
    result = result.sort_values(by=['ticker', 'date']).reset_index(drop=True)    

    time_span = (datetime.datetime.now() - datetime.datetime(2022, 6, 14)).days
    print(time_span)
    last_dates = list(result.date.drop_duplicates())[-time_span:]
    last_dates = last_dates[last_dates.index('2022-06-14'):]
    print(last_dates)
    result.to_csv('pct1_factors/LGBMRegressor_predict_value'+last_dates[-1]+'.csv', index = False)
    for index in range(len(last_dates)):
        last_date = last_dates[index]
        print(last_date)
        mydate = pd.to_datetime(last_date)
        if datetime.datetime.now() >= mydate and mydate >= datetime.datetime(2022, 6, 14):
            temp_result = result[result.date==last_date].copy()
            temp_date = mydate.strftime("%Y%m%d")
            temp_result['date'] = temp_date
            temp_result.to_csv('pct1_factors/prediction_results/'+temp_date+'.csv', index = False)
            #print(temp_result)
    #p_date = list(pd.to_datetime(result.date).drop_duplicates())[-1].strftime("%Y%m%d")
    #result = result[result.date ==last_date ]
    #result['date'] = p_date
    #result.to_csv('pct1_factors/prediction_results'+today+'.csv', index = False)
    #result.to_csv('pct1_factors/LGBMRegressor_predict_value'+today+'.csv')

    #pickle.dump(pred, open("pred", "wb"))