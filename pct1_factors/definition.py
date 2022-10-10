import numpy as np

unary_list = ['abs', 'log', 'sign', 'sqrt', 'square', 'relu', 'sigmoid', 'rank',
              'ind_neutralize', 'ind_rank']

ts_list = ['sum', 'wma', 'tsmax', 'tsmin', 'delay', 'std', 'delta', 'tsrank', 'median',
           'highday', 'lowday', 'mean', 'decaylinear', 'var', 'skewness', 'kurtosis']

binary_list = ['min', 'max', '+', '-', '*', '/', 'sub_rank', 'div_rank']

bi_ts_list = ['corr', 'covariance']

all_func_list = ['abs', 'log', 'sign', 'sqrt', 'square', 'relu', 'sigmoid', 'rank',
                 'sum', 'wma', 'tsmax', 'tsmin', 'delay', 'median', 'tsrank',
                 'std', 'delta', 'highday', 'lowday', 'mean',
                 'decaylinear', 'var', 'skewness', 'kurtosis',
                 'ind_neutralize', 'ind_rank',
                 'min', 'max', 'min', 'max',
                 '+', '-', '*', '/',
                 '+', '-', '*', '/',
                 'sub_rank', 'div_rank',
                 'corr', 'covariance',
                 'corr', 'covariance']

op_child_num = {}
for op in unary_list:
    op_child_num[op] = 1
for op in binary_list + ts_list:
    op_child_num[op] = 2
for op in bi_ts_list:
    op_child_num[op] = 3

max_range = 30
min_range = 15
op_const_range = [i for i in range(min_range, max_range+1, 5)]

price_variables = ['open', 'high', 'low', 'close', 'vwap', 'MA5', 'EMA5',
                   'popen', 'phigh', 'plow', 'pclose', 'pvwap']
volume_variables = ['volume', 'amount', 'OBV20']
ratio_variables = ['ret', 'turnover', 'KDJ', 'CCI5', 'DIF', 'RSI', 'MACD']
other_variables = ['PB', 'PE', 'LCAP', 'BETA', 'MOMENTUM', 'RESVOL', 'LIQUIDITY']

op_remain_original = ['abs', 'log', 'sqrt', 'square', 'relu', 'sum', 'wma', 'tsmax', 'tsmin', 'delay', 'delta',
                      'median', 'mean', 'decaylinear', 'min', 'max', '+', '-']
op_become_ratio = ['sign', 'sigmoid', 'rank', 'std', 'tsrank', 'highday', 'lowday', 'var', 'sub_rank',
                   'div_rank', 'corr', 'covariance', 'skewness', 'kurtosis', 'ind_neutralize', 'ind_rank']
op_special = ['*', '/']
# inv没处理


class node:
    def __init__(self, name, children):
        self.name = name
        self.children = children
        self.data = None

    def do_deletion(self):
        self.data = None
        for child in self.children:
            child.do_deletion()

    def do_partial_deletion(self, depth):
        if depth == 0:
            self.do_deletion()
        else:
            for child in self.children:
                child.do_partial_deletion(depth - 1)


class paramnode:
    def __init__(self, name):
        self.name = name
        self.data = None

    def do_deletion(self):
        self.data = None

    def do_partial_deletion(self, depth):
        if depth == 0:
            self.do_deletion()


class constnode:
    def __init__(self, value):
        self.name = value
        self.data = self.name

    def do_deletion(self):
        pass

    def do_partial_deletion(self, depth):
        pass

    def change_value(self, value):
        self.name = value
        self.data = value

