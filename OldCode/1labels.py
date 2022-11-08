import pandas as pd

filenames = ['label_askbid_pct1.csv', 'label_openclose_pct1.csv']
for file in filenames:
    file = '/data/liufengyuan/raw_data/' + file
    name = file[33:-4]
    print(name)
    label = name + '_rank'
    print(label)
    alpha = pd.read_csv(file)
    alpha = alpha.loc[alpha['date'] >= '2017-01-03']
    alpha = alpha.rename(columns = lambda x: x[:-5] if (x != 'date') else 'tradeDate')
    alpha = alpha.set_index(['tradeDate']).stack().reset_index().rename(columns={'level_1': 'ticker', 0: name})
    alpha['ticker'] = alpha['ticker'].astype(int)
    alpha = alpha.sort_values(by=['ticker', 'tradeDate']).reset_index(drop=True)
    alpha[label] = alpha.groupby('tradeDate')[name].rank(pct = True)
    print(alpha)
    #alpha.to_csv('/data/liufengyuan/labels/{}.csv'.format(name), index=False)
    print('----- Finished FILE {}------'.format(file))