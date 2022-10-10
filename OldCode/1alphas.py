import glob
import pandas as pd
from joblib import Parallel, delayed
import time

path1 = 'raw_data/alphas_036'
filenames = glob.glob(path1 + '/*.csv')
result = pd.DataFrame()
for file in filenames[:5]:
    i = 1
    dic_name = file[9:19]
    print(dic_name)
    name = dic_name[7:] + '_' +file[20:-4]
    print(name)
    alpha = pd.read_csv(file)
    alpha = alpha.loc[alpha['date'] >= '2017-01-03']
    if dic_name == 'alphas_036':
        alpha = alpha.rename(columns = lambda x: x[2:] if (x != 'date') else 'tradeDate')
    else:
        alpha = alpha.rename(columns={"date": "tradeDate"})
    alpha = alpha.set_index(['tradeDate']).stack().reset_index().rename(columns={'level_1': 'ticker', 0: name})
    alpha['ticker'] = alpha['ticker'].astype(int)
    alpha = alpha.set_index(['ticker', 'tradeDate'])
    alpha = alpha.sort_index(ascending=True)
    if result.size == 0:
        print('result is empty')
        result = alpha.copy()
    else:
        result = pd.concat([result, alpha], axis=1)
    print('----- Finished FILE {}------'.format(file))
print(result)
#result.to_csv('alphas_066.csv', index=False)