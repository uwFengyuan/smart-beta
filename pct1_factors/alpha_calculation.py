import numpy as np
from definition import *
from get_formula import formula_to_tree
from copy import deepcopy


class Alpha_Calculation():
    def __init__(self):
        pass

    def set_industry_info(self, info):
        self.industry_info = info

    def get_paramnode(self, tree):
        list = []
        if isinstance(tree, paramnode):
            return [tree]
        elif isinstance(tree, constnode):
            return []
        elif isinstance(tree, node):
            for children in tree.children:
                result = self.get_paramnode(children)
                list.extend(result)
        else:
            print("We cannot recognize tree type:", type(tree))
            exit()
        return list

    def calculate_features(self, tree, data, var_names):
        if isinstance(tree, str):
            tree = formula_to_tree(tree, var_names)
        if tree.data is None:
            paramnode_list = self.get_paramnode(tree)
            for leaf in paramnode_list:
                leaf.data = data[leaf.name]
            if tree.data is None:
                tree.data = self.do_calculation(tree)
                tree.data.replace([np.inf, -np.inf], np.nan, inplace=True)

    def calculate_features_cnt(self, tree):
        tree.data = self.do_calculation(tree)
        tree.data.replace([np.inf, -np.inf], np.nan, inplace=True)


    def calculate_fitness(self, tree, label, method):
        if method == 'IC':
            return self.calculate_IC(tree, label, method='pearson')
        elif method == 'rankIC':
            return self.calculate_IC(tree, label, method='spearman')
        elif method == 'ret':
            return self.calculate_ret(tree, label)
        elif method == 'ICIR':
            return self.calculate_ICIR(tree, label)
        else:
            raise NotImplementedError(f"Cannot recognize method {method}.")

    def calculate_IC(self, tree, label, method, threshold):
        tree.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data_tem = deepcopy(tree.data.dropna(axis='index', how='all'))
        label_tem = deepcopy(label.reindex(data_tem.index))
        if len(data_tem) == 0:
            corr = 0
        elif np.mean(np.mean(np.isnan(data_tem))) >= threshold:
            corr = 0
        else:
            corr = data_tem.corrwith(label_tem, axis='columns', method=method)
            if np.mean(np.isnan(corr)) > threshold:
                corr = 0
            else:
                corr = np.mean(corr)
            if ~np.isfinite(corr):
                corr = 0
            else:
                pass
        del data_tem
        del label_tem
        return corr

    def calculate_ICIR(self, tree, label, threshold):
        tree.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data_tem = deepcopy(tree.data.dropna(axis='index', how='all'))
        label_tem = deepcopy(label.reindex(data_tem.index))
        if len(data_tem) == 0:
            ICIR = 0
        elif np.mean(np.mean(np.isnan(data_tem))) >= threshold:
            ICIR = 0
        else:
            corr = data_tem.corrwith(label_tem, axis='columns')
            ICIR = np.mean(corr) / np.std(corr)
            if ~np.isfinite(ICIR):
                ICIR = 0
            else:
                pass
        del data_tem
        del label_tem
        return ICIR

    def calculate_ret(self, tree, label, top_n):
        if self.calculate_IC(tree, label) < 0:
            behavior = self.get_top_n_boolean_dataframe(-tree.data, top_n)
        else:
            behavior = self.get_top_n_boolean_dataframe(tree.data, top_n)
        ret = label[behavior].mean(axis=1).dropna().mean()
        if np.isfinite(ret):
            return ret
        else:
            return -np.inf

    def get_top_n_boolean_dataframe(self, data, top_n=30):
        def _func(x):
            idx = x.nlargest(top_n).index
            x[x.index] = False
            x[idx] = True
            return x
        boolean_df = deepcopy(data.dropna(how='all', axis='index'))
        boolean_df = boolean_df.apply(_func, axis=1)
        return boolean_df

    def do_calculation(self, tree):
        for child in tree.children:
            if child.data is None:
                child.data = self.do_calculation(child)

        if tree.name == 'abs':
            return tree.children[0].data.abs()

        elif tree.name == 'log':
            return np.log(np.abs(tree.children[0].data))

        elif tree.name == 'sign':
            return np.sign(tree.children[0].data)

        elif tree.name == 'sqrt':
            return np.sqrt(np.abs(tree.children[0].data))

        elif tree.name == 'square':
            return np.square(tree.children[0].data)

        elif tree.name == 'cube':
            return np.power(tree.children[0].data, 3)

        elif tree.name == 'ind_neutralize':
            data = deepcopy(tree.children[0].data)
            for key in self.industry_info:
                if len(self.industry_info[key]) == 1:
                    continue
                avg = tree.children[0].data.loc[:, self.industry_info[key]].mean(axis=1, numeric_only=True)
                data.loc[:, self.industry_info[key]] = tree.children[0].data.loc[:, self.industry_info[key]].sub(avg,
                                                                                                             axis='index')
            return data

        elif tree.name == 'ind_rank':
            data = deepcopy(tree.children[0].data)
            for key in self.industry_info:
                data.loc[:, self.industry_info[key]] = \
                    tree.children[0].data.loc[:, self.industry_info[key]].rank(axis=1, numeric_only=True, ascending=True, pct=True)
            return data

        elif tree.name == 'sigmoid':
            return 1 / (1 + np.exp(-tree.children[0].data))

        elif tree.name == 'relu':
            data_tem = tree.children[0].data
            data_tem[data_tem < 0] = 0
            return data_tem

        elif tree.name == 'rank':
            return tree.children[0].data.rank(axis=1, numeric_only=True, ascending=True, pct=True)

        elif tree.name == 'inv':
            return 1 / tree.children[0].data

        # -----------------ts_list--------------------#
        elif tree.name == 'sum':
            window = tree.children[1].data
            return tree.children[0].data.rolling(window, min_periods=op_const_range[0]).sum()

        elif tree.name == 'wma':
            window = tree.children[1].data
            weight = np.power(0.9, np.arange(start=window, stop=0, step=-1))
            _wma = lambda x: np.nansum(x * weight)
            return tree.children[0].data.rolling(window).apply(_wma, raw=True)

        elif tree.name == 'tsmax':
            window = tree.children[1].data
            return tree.children[0].data.rolling(window, min_periods=op_const_range[0]).max()

        elif tree.name == 'tsmin':
            window = tree.children[1].data
            return tree.children[0].data.rolling(window, min_periods=op_const_range[0]).min()

        elif tree.name == 'delay':
            window = tree.children[1].data
            return tree.children[0].data.shift(window)

        elif tree.name == 'std':
            window = tree.children[1].data
            return tree.children[0].data.rolling(window, min_periods=op_const_range[0]).std()

        elif tree.name == 'delta':
            window = tree.children[1].data
            return tree.children[0].data - tree.children[0].data.shift(window)

        elif tree.name == 'tsrank':
            window = tree.children[1].data
            _rank = lambda x: (x <= (np.array(x))[-1]).sum() / float(x.shape[0])
            return tree.children[0].data.rolling(window, min_periods=op_const_range[0]).apply(_rank, raw=True)

        elif tree.name == 'highday':
            window = tree.children[1].data

            def rolling_highday(df):
                return (window - 1 - np.argmax(df)) / window

            return tree.children[0].data.rolling(window).apply(rolling_highday, raw=True)

        elif tree.name == 'lowday':
            window = tree.children[1].data

            def rolling_lowday(df):
                return (window - 1 - np.argmin(df)) / window

            return tree.children[0].data.rolling(window).apply(rolling_lowday, raw=True)

        elif tree.name == 'mean':
            window = tree.children[1].data
            return tree.children[0].data.rolling(window, min_periods=op_const_range[0]).mean()

        elif tree.name == 'decaylinear':
            window = tree.children[1].data
            dividend = window * (window + 1) / 2
            weights = (np.arange(window) + 1) / dividend

            def _decay_linear_avg(df):
                return np.nansum(df * weights)

            return tree.children[0].data.rolling(window).apply(_decay_linear_avg, raw=True)

        elif tree.name == 'median':
            window = tree.children[1].data
            return tree.children[0].data.rolling(window, min_periods=op_const_range[0]).median()

        elif tree.name == 'var':
            window = tree.children[1].data
            return tree.children[0].data.rolling(window, min_periods=op_const_range[0]).var()

        elif tree.name == 'skewness':
            window = tree.children[1].data
            return tree.children[0].data.rolling(window, min_periods=op_const_range[0]).skew()

        elif tree.name == 'kurtosis':
            window = tree.children[1].data
            return tree.children[0].data.rolling(window, min_periods=op_const_range[0]).kurt()

        # ------------------------binary_list-----------------------------#
        elif tree.name == 'max':
            return np.maximum(tree.children[0].data, tree.children[1].data)

        elif tree.name == 'min':
            return np.minimum(tree.children[0].data, tree.children[1].data)

        elif tree.name == '+':
            return tree.children[0].data + tree.children[1].data

        elif tree.name == '-':
            return tree.children[0].data - tree.children[1].data

        elif tree.name == '*':
            return tree.children[0].data * tree.children[1].data

        elif tree.name == '/':
            return tree.children[0].data / tree.children[1].data

        elif tree.name == 'sub_rank':
            return tree.children[0].data.rank(axis=1, numeric_only=True, ascending=True, pct=True) - tree.children[1].data.rank(axis=1, numeric_only=True, ascending=True, pct=True)

        elif tree.name == 'div_rank':
            return tree.children[0].data.rank(axis=1, numeric_only=True, ascending=True, pct=True) / tree.children[1].data.rank(axis=1, numeric_only=True, ascending=True, pct=True)

        # -------------------------bi_ts_list------------------------#
        elif tree.name == 'corr':
            window = tree.children[2].data
            return tree.children[0].data.rolling(window, min_periods=op_const_range[0]).corr(tree.children[1].data)

        elif tree.name == 'covariance':
            window = tree.children[2].data
            return tree.children[0].data.rolling(window, min_periods=op_const_range[0]).cov(tree.children[1].data)

        else:
            print("Cannot recognize tree.name: %s" % (tree.name))
            exit()
