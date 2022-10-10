import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from definition import unary_list, binary_list, ts_list, bi_ts_list, constnode, op_const_range, node, paramnode
from random import choice, random


def _myload(path_data, start_date, end_date, label_name, var_name, columns):
    if (var_name != label_name) and ('label' in var_name):
        return None, None

    var = pd.read_csv(os.path.join(path_data, '%s.csv' % var_name), index_col=0)
    var.index.name = 'date'
    var.index = [str(item) for item in var.index]
    var.index = pd.to_datetime(var.index).strftime("%Y-%m-%d")
    var.index.name = 'date'
    var.sort_index(inplace=True)
    # if var.shape != (2994,4801):
    #     print(var_name, "has something wrong.", var.shape)
    #     return None, None
    if columns is not None:
        var = var[columns]
    if start_date is not None:
        var = var.loc[var.index[var.index >= start_date]]
    if end_date is not None:
        var = var.loc[var.index[var.index <= end_date]]
    return var_name, var


def load_data(path_data, start_date=None, end_date=None, label_name='label', var_list=None,
              columns=None, reindex=False):
    print("Start loading data.")
    start = datetime.now()
    data = {}
    full_files = os.listdir(path_data)
    if var_list is None:
        var_list = [item[:-4] for item in full_files]
    elif label_name not in var_list:
            var_list.append(label_name)
    ex = ProcessPoolExecutor(40)
    results = []
    for var_name in var_list:
        results.append(ex.submit(_myload, path_data, start_date, end_date, label_name, var_name, columns, ))
    ex.shutdown(wait=True)
    for res in results:
        var_name, var = res.result()
        if var_name is None:
            continue
        if 'label' in var_name:
            data['label'] = var
        else:
            data[var_name] = var
    if reindex:
        for key in data:
            data[key] = data[key].loc[data['label'].index]
    print("Time spent loading data:", datetime.now() - start)
    return data


def load_industry_info(label, path, index_col='secID'):
    ind_info = pd.read_csv(path, encoding="gbk")
    ind_info = ind_info.set_index(index_col)
    ind_info.index = ind_info.index.astype(str)
    ind_info = ind_info.loc[list(label.columns)]
    ind_dict = {}
    for key in ind_info['industryName1'].unique():
        ind_dict[key] = []
    for idx in ind_info.index:
        ind_dict[ind_info['industryName1'].loc[idx]].append(idx)
    return ind_dict


def make_random_tree(maxdepth, var_names, fpr=0.9, flag=1):
    if maxdepth > 0 and (flag == 1 or random() < fpr):
        children = []
        op = choice(unary_list + ts_list*2 + binary_list*2 + bi_ts_list*2)
        if op in ts_list:
            children.append(make_random_tree(maxdepth - 1, var_names, fpr, 1))
            children.append(constnode(choice(op_const_range)))
        elif op in bi_ts_list:
            children.append(make_random_tree(maxdepth - 1, var_names, fpr, 1))
            children.append(make_random_tree(maxdepth - 1, var_names, fpr, 0))
            children.append(constnode(choice(op_const_range)))
        elif op in unary_list:
            children = [make_random_tree(maxdepth - 1, var_names, fpr, 1)]
        elif op in binary_list:
            children.append(make_random_tree(maxdepth - 1, var_names, fpr, 1))
            children.append(make_random_tree(maxdepth - 1, var_names, fpr, 0))
        else:
            print(f"Cannot recognize {op} in make_random_tree!")
            exit()
        return node(op,children)
    else:
        return paramnode(choice(var_names))