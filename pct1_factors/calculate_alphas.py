from utils import load_data,  load_industry_info
from concurrent.futures import ProcessPoolExecutor
from alpha_calculation import Alpha_Calculation
from get_formula import formula_to_tree, tree_to_formula
import traceback
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int)
parser.add_argument("--threshold", type=float)
args = parser.parse_args()





def calculation(tree):
    try:
        if isinstance(tree, str):
            tree = formula_to_tree(tree, var_names)
        Calculator = Alpha_Calculation()
        Calculator.set_industry_info(ind_info)
        Calculator.calculate_features(tree, data, var_names)
        # tree.data = tree.data.rank(axis=1, numeric_only=True, ascending=True, pct=True)
        tree.do_partial_deletion(depth=1)
        return tree
    except Exception as e:
        if isinstance(tree, str):
            print(tree)
        else:
            print(tree_to_formula(tree))
        print(traceback.format_exc())


if __name__ == '__main__':
    # zz1000 = pd.read_csv('/home/zhangtianping/AutoAlpha3/MS/DataManagement/zz1000_cons/2020-12-31.csv', encoding="gbk")
    # zz1000 = zz1000['ticker'].to_list()

    data = load_data('/workspace1/liufengyuan/pct1_factors/data/',label_name = 'label_fake')
    print(data)
    label = data['label']
    ind_info = load_industry_info(label, path='/workspace1/liufengyuan/pct1_factors/industry_info_xtech.csv', index_col='ticker')
    all_index = label.stack(dropna=False).index
    del data['label']
    var_names = list(data.keys())
    population = []


    new_features = ['max(var((gaopin_liangjia_FACTORS_c19*minute_FACTORS_RSJ),15),sigmoid(sub_rank(gaopin_liangjia_FACTORS_c79,gaopin_liangjia_FACTORS_c78)))', 'rank(((uq_liangjia_FACTORS_ACD6/gaopin_liangjia_FACTORS_c71)+wma(winrate_FACTORS_prob,15)))', 'lowday(relu(max(uq_liangjia_FACTORS_LCAP,gaopin_liangjia_FACTORS_c20)),15)', 'rank(lowday(tsmin(liangjia_FACTORS_BIAS60,20),30))', 'mean(ind_rank(sigmoid(choice_FACTORS_HQFW_SIGNAL)),25)', 'delay(kurtosis(highday(fund_FACTORS_num,25),20),15)', 'rank(wma(sign(minute_new_FACTORS_ILLIQ),20))', 'rank(tsmin((second_liangjia_FACTORS_lag-second_liangjia_FACTORS_lag),15))', 'sum(ind_rank(div_rank(choice_FACTORS_HQFW_SIGNAL,choice_FACTORS_HQFW_SIGNAL)),30)', 'rank(tsrank(sign(any_cor_FACTORS_mom_cl),25))', 'min(std(covariance(gaopin_liangjia_FACTORS_c125,gaopin_liangjia_FACTORS_c78,20),20),sub_rank(abs(gaopin_liangjia_FACTORS_c80),gaopin_liangjia_FACTORS_c77))', '(lowday(sqrt(uq_liangjia_FACTORS_LCAP),15)*sum(div_rank(uq_liangjia_FACTORS_LFLO,minute_FACTORS_adj_mom),20))', 'rank(min(ind_rank(second_liangjia_FACTORS_lag),gaopin_liangjia_FACTORS_c22))', 'ind_neutralize(sigmoid(delay(gaopin_liangjia_FACTORS_c13,20)))', 'rank((ind_rank(minute_FACTORS_weipan)*(gaopin_liangjia_FACTORS_c20+gaopin_liangjia_FACTORS_c22)))', '(min(std(gaopin_liangjia_FACTORS_c14,25),square(gaopin_liangjia_FACTORS_c125))/(sqrt(jq_liangjia_FACTORS_VSTD10)*highday(uq_liangjia_FACTORS_LCAP,25)))', '(decaylinear(sqrt(winrate_FACTORS_prob),30)*minute_FACTORS_arpp)', 'rank(tsrank(max(gaopin_liangjia_FACTORS_c69,uq_liangjia_FACTORS_LFLO),25))', 'rank(sigmoid((uq_liangjia_FACTORS_MTM*gaopin_liangjia_FACTORS_c30)))', '(lowday(tsmin(uq_liangjia_FACTORS_LCAP,15),30)/square(var(minute_new_FACTORS_TMA_turn,30)))', 'tsrank(sub_rank((gaopin_liangjia_FACTORS_c79+gaopin_liangjia_FACTORS_c14),max(minute_new_FACTORS_TMA_turn,gaopin_liangjia_FACTORS_c77)),30)', '(covariance(tsmax(gaopin_liangjia_FACTORS_c80,15),delta(gaopin_liangjia_FACTORS_c77,20),15)+tsrank(max(gaopin_liangjia_FACTORS_c71,gaopin_liangjia_FACTORS_c125),15))', 'div_rank((sum(gaopin_liangjia_FACTORS_c80,20)/gaopin_liangjia_FACTORS_c20),median(var(winrate_FACTORS_prob,20),30))', 'delta(relu(sub_rank(gaopin_liangjia_FACTORS_c77,gaopin_liangjia_FACTORS_c107)),30)', 'rank(tsrank(sign(gaopin_liangjia_FACTORS_c22),20))', 'rank(tsmin(sign(gaopin_liangjia_FACTORS_c30),15))', '(min(sub_rank(gaopin_liangjia_FACTORS_c71,gaopin_liangjia_FACTORS_c69),tsmax(any_cor_FACTORS_mom_cl,20))+tsrank(sqrt(uq_liangjia_FACTORS_LCAP),15))', 'rank(std(sign(gaopin_liangjia_FACTORS_c69),20))', 'min(square(tsrank(gaopin_liangjia_FACTORS_c107,30)),min((gaopin_liangjia_FACTORS_c77-gaopin_liangjia_FACTORS_c80),var(choice_FACTORS_HQFW_SIGNAL,20)))', '(tsrank(sqrt(gaopin_liangjia_FACTORS_c71),15)+min(square(gaopin_liangjia_FACTORS_c78),rank(jq_liangjia_FACTORS_boll_down)))', 'rank(sigmoid((L2_FACTORS_Stren_42/minute_new_FACTORS_TMA_turn)))', 'rank(delta(sum(winrate_FACTORS_prob,20),15))', 'ind_neutralize(((uq_liangjia_FACTORS_LCAP/x_tech_liangjia_FACTORS_factor_model_genetic_028)-tsmin(gaopin_liangjia_FACTORS_c22,20)))', '(relu((gaopin_liangjia_FACTORS_c20/gaopin_liangjia_FACTORS_c79))*winrate_FACTORS_prob)', '(var((liangjia_FACTORS_BIAS60-second_liangjia_FACTORS_bear),20)*((gaopin_liangjia_FACTORS_c78+liangjia_FACTORS_BIAS60)+skewness(gaopin_liangjia_FACTORS_c120,30)))', 'max(abs(lowday(fund_FACTORS_num,15)),min(delay(suntime_FACTORS_market_confidence_5d,25),sum(minute_new_FACTORS_ILLIQ,20)))', '(sign((minute_new_FACTORS_IMI+uq_liangjia_FACTORS_ACD6))*sqrt(max(winrate_FACTORS_prob,minute_FACTORS_arpp)))', 'abs(covariance(sigmoid(uq_liangjia_FACTORS_LFLO),uq_liangjia_FACTORS_LFLO,20))', 'ind_rank(sign(tsmax(minute_new_FACTORS_TMA_turn,20)))', 'rank(tsmin(tsmin(gaopin_liangjia_FACTORS_c71,25),20))', 'ind_rank(sign(abs(liangjia_FACTORS_Rank1M)))', 'rank(square(tsrank(gaopin_liangjia_FACTORS_c22,20)))', 'ind_rank(var(sign(choice_FACTORS_HQFW_SIGNAL),20))', 'rank(lowday(std(fund_FACTORS_num,25),15))', '(sub_rank(sign(x_tech_liangjia_FACTORS_factor_model_genetic_028),sign(gaopin_liangjia_FACTORS_c125))/sign(max(gaopin_liangjia_FACTORS_c120,minute_FACTORS_arpp)))', 'std(lowday(max(fund_FACTORS_num,uq_liangjia_FACTORS_MTM),15),15)', 'std(sub_rank(sigmoid(gaopin_liangjia_FACTORS_c14),sign(gaopin_liangjia_FACTORS_c20)),25)', 'ind_rank(lowday(sign(choice_FACTORS_HQFW_SIGNAL),30))', 'sqrt(div_rank(sign(uq_liangjia_FACTORS_MTM),(x_tech_liangjia_FACTORS_ADX/x_tech_liangjia_FACTORS_ADX)))', '(sum(tsmax(gaopin_liangjia_FACTORS_c13,20),20)*((liangjia_FACTORS_BIAS60+uq_liangjia_FACTORS_ACD6)*std(any_cor_FACTORS_mom_cl,20)))', 'sqrt((square(gaopin_liangjia_FACTORS_c14)/median(gaopin_liangjia_FACTORS_c71,30)))', 'decaylinear(rank(sign(gaopin_liangjia_FACTORS_c78)),25)', 'ind_rank(mean(sign(gaopin_liangjia_FACTORS_c13),30))', 'square(max(div_rank(gaopin_liangjia_FACTORS_c69,gaopin_liangjia_FACTORS_c71),(liangjia_FACTORS_Rank1M+L2_FACTORS_f2)))', 'div_rank(tsrank(tsmax(uq_liangjia_FACTORS_LFLO,20),30),tsmin(highday(gaopin_liangjia_FACTORS_c79,15),30))', 'max((sigmoid(uq_liangjia_FACTORS_LFLO)+tsrank(gaopin_liangjia_FACTORS_c14,15)),min(tsrank(uq_liangjia_FACTORS_ACD6,25),delta(gaopin_liangjia_FACTORS_c125,25)))', 'rank(sign(ind_neutralize(gaopin_liangjia_FACTORS_c105)))', 'rank(sign(lowday(gaopin_liangjia_FACTORS_c79,30)))', 'ind_rank(delay(sign(choice_FACTORS_HQFW_SIGNAL),20))', '(kurtosis(median(jq_liangjia_FACTORS_boll_down,30),30)+((second_liangjia_FACTORS_bear+minute_FACTORS_weipan)*(gaopin_liangjia_FACTORS_c22-jq_liangjia_FACTORS_boll_down)))', 'ind_rank(sign(tsrank(uq_liangjia_FACTORS_ACD6,20)))', 'ind_rank(std(sign(jq_liangjia_FACTORS_boll_down),20))', 'tsrank(max(min(gaopin_liangjia_FACTORS_c13,gaopin_liangjia_FACTORS_c14),sign(second_liangjia_FACTORS_lag)),30)', 'ind_rank(sign(tsrank(uq_liangjia_FACTORS_MTM,20)))', 'kurtosis(delay(tsrank(fund_FACTORS_num,30),20),15)', 'rank(sign(lowday(gaopin_liangjia_FACTORS_c77,15)))']

    print(len(new_features))
    results = []
    ex = ProcessPoolExecutor(40)
    for tree in new_features:
        results.append(ex.submit(calculation, tree))
    ex.shutdown(wait=True)
    count = 0
    for res in results:
        tree = res.result()
        print(count, tree_to_formula(tree))
        tree.data.to_csv('/workspace1/liufengyuan/pct1_factors/data/alpha_%d.csv' % count)
        count += 1


