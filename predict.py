import lightgbm as lgb
assert lgb.__version__ == '3.3.2'
import numpy as np
import pandas as pd
import time

f_index = ['ticker', 'date']
def get_data():
    # 这个是做数据的columns的顺序，需要和训练时对齐
    start = time.time()
    a = np.load('./feature_order.npy')
    alpha_list = []
    for num in range(0, 66):
        alpha_list.append('alpha_{}'.format(num))
    f_x = a.tolist()[:-66] + alpha_list
    result_list = []
    for name in f_x:
        print(name)
        if name == 'x_tech_liangjia_FACTORS_MF_20':
            result_list[-1][name] = np.nan
        elif name == 'minute_new_FACTORS_CDPDP':
            result_list[-1][name] = np.nan
        elif name == 'suntime_FACTORS_total_score':
            result_list[-1][name] = np.nan
        else:
            alpha = pd.read_csv('data/' + name + '.csv')
            if alpha.columns.values.tolist()[0] == 'tradeDate':
                alpha = alpha.rename(columns={"tradeDate": "date"})
            alpha = alpha.set_index(['date']).stack().reset_index().rename(columns={'level_1': 'ticker', 0: name})
            alpha['ticker'] = alpha['ticker'].astype(int)
            alpha = alpha.set_index(f_index)
            alpha = alpha.sort_index(ascending=True)
            result_list.append(alpha)
    result = pd.concat(result_list, axis=1)
    print(f'耗时:{time.time() - start}')
    print('======== COMPLETED ========')
    result.to_csv('predict_factors.csv')
    return result

if __name__ == '__main__':
    import pickle
    save_path = './models/'
    models = []
    for i in range(10):
        file_name = save_path + 'gbm_model_%d.pickle' % i
        with open(file_name, 'rb') as file:
            gbm = pickle.load(file)
            models.append(gbm)

    data = get_data() #pd.read_csv('predict_factors.csv') #
    print(data)
    #data = data.sort_values(by=['ticker', 'date']).reset_index(drop=True)
    #data = data.set_index(['ticker', 'date'])
    pred = np.zeros(data.shape[0])
    for gbm in models:
        pred += gbm.predict(data.values) / len(models)
    result = pd.DataFrame(index=data.index)
    result['prediction'] = pred
    result.to_csv('LGBMRegressor_predict_value.csv', index = False)
    print(result)
    #pickle.dump(pred, open("pred", "wb"))