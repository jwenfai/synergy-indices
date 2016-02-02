__author__ = 'jlow@masdar.ac.ae'

import numpy as np
import pandas as pd
import re
import harsanyi as har
import grabisch as gra
import graph_generator as gg
import fortransetops as fso
import scipy.stats as stats
import time


def find_avg_impact(N_):
    all_agents = np.unique(np.hstack([list(group_) for group_ in N_[:, 0]]))
    agent_dict = dict(zip(all_agents, np.arange(len(all_agents))))
    score_list = [[] for x in range(len(all_agents))]
    for group_, score_ in N_:
        for agent_ in group_:
            score_list[agent_dict[agent_]].append(score_/float(len(group_)))
    return np.array([np.mean(scores_) for scores_ in score_list]), agent_dict


def find_size_avg_impact_nonadd_v3(N_):
    all_agents = np.unique(np.hstack([list(group_) for group_ in N_[:, 0]]))
    agent_dict = dict(zip(all_agents, np.arange(len(all_agents))))
    all_sizes = np.array([len(group_) for group_ in N_[:, 0]])
    size_mean, size_std,  = all_sizes.mean(), all_sizes.std()
    score_mean, score_std = N_[:, 1].mean(), N_[:, 1].std()
    score_list = [[] for x in range(len(all_agents))]
    size_list = [[] for x in range(len(all_agents))]
    normed_scores = ((N_[:, 1] - score_mean) / score_std) * (1+((all_sizes - size_mean) / size_std))
    for group_, score_ in zip(N_[:, 0], normed_scores):
        for agent_ in group_:
            group_size = float(len(group_))
            score_list[agent_dict[agent_]].append(score_/group_size)
            size_list[agent_dict[agent_]].append(group_size)
    agent_avimp = np.array([np.mean(scores_) for scores_ in score_list])
    agent_sizes = np.array([np.mean(sizes_) for sizes_ in size_list])
    return agent_avimp, agent_sizes, agent_dict, np.vstack([N_[:, 0], normed_scores]).transpose()


def find_synergy(avimp, agent_dict, N_):
    synergy = []
    for group_, score_ in N_:
        total_impact = 0
        for agent_ in group_:
            total_impact += avimp[agent_dict[agent_]]
        synergy.append(score_ - total_impact)
    return np.array(synergy)


def synergy_index(random_coalitions, N_bool, N_scores):
    agents_avg_shap = gg.avg_shap(N_bool, N_scores)
    coalition_values = []
    synergy_values = []
    random_coalitions_issubset = fso.is_subset(random_coalitions, N_bool, len(random_coalitions), len(N_bool), len(N_bool[0])).astype(bool)
    print('Subset boolean generated.')
    for ix in range(len(random_coalitions)):
        avg_shap_sum = np.sum(agents_avg_shap[random_coalitions[ix].astype(bool)])
        coalition_val = N_scores[random_coalitions_issubset[ix]].sum()
        synergy_val = coalition_val - avg_shap_sum
        coalition_values.append(coalition_val)
        synergy_values.append(synergy_val)
    return synergy_values, coalition_values, agents_avg_shap


def save_agent_dict(agents_dict, filepath_):
    pd.DataFrame([[key_, agents_dict[key_]] for key_ in sorted(agents_dict.keys())], columns=['key', 'value']).to_csv(filepath_, index=False, sep='\t')


results_data_path = '/Users/huhwhat/Documents/synergy-indices/ResultsData/'
runtimes_file = open(results_data_path+'runtimes.csv', 'w+')
runtimes_file.write('domain,index,runtime,coalitions,agents\n')
n_coalitions = []


####################################
## Load real dataset here: Drinks ##
####################################
domain_id = 0
domain_name = 'drinks'
print(domain_name)
N_df = pd.DataFrame.from_csv('/Users/huhwhat/Documents/synergy-indices/Datasets/absolut_drinks/ingredients_rating.csv', index_col=None)
N_df['rating'] = N_df['rating'].astype(float)
N_ = np.array([[set(re.split(',', obsv[0])), obsv[1]] for obsv in N_df.values]).astype(object)
N_ = N_[N_[:, 1] <= 100]  # Removes a possible outlier (the only drink with a rating > 100)
N_[:, 1] -= 50.  # normalization/rescaling since this is a non-additive domain
N_with_duplicates = N_.copy()
N_bool_duplicates, N_scores_duplicates, all_agents_dict_duplicates = gg.make_bool_N(N_with_duplicates)
N_ = gg.duplicates_to_mean(N_)
N_bool, N_scores, all_agents_dict = gg.make_bool_N(N_)

n_coalitions.append([domain_name, len(N_bool), len(N_bool_duplicates)])
synergy_values, coalition_values, agents_avg_shap = synergy_index(N_bool, N_bool, N_scores)
start = time.time()
synergy_values_duplicates, coalition_values_duplicates, agents_avg_shap_duplicates = synergy_index(N_bool_duplicates, N_bool, N_scores)
stop = time.time()
runtimes_file.write(','.join([domain_name, 'SI', str(stop-start)] + [str(x) for x in N_bool.shape]) + '\n')
har_syn = har.find_synergy(N_bool, N_bool, N_scores)
start = time.time()
har_syn_duplicates = har.find_synergy(N_bool_duplicates, N_bool, N_scores)
stop = time.time()
runtimes_file.write(','.join([domain_name, 'HA', str(stop-start)] + [str(x) for x in N_bool.shape]) + '\n')
start = time.time()
gra_syn = gra.find_synergy(N_bool, N_bool, N_scores)
gra_syn_duplicates = gra.find_synergy(N_bool_duplicates, N_bool, N_scores)
stop = time.time()
runtimes_file.write(','.join([domain_name, 'GR', str(stop-start)] + [str(x) for x in N_bool.shape]) + '\n')
# simple_avimp, simple_agent_dict = find_avg_impact(N_)
# simple_synergy = find_synergy(simple_avimp, simple_agent_dict, N_)
# simple_avimp_duplicates, simple_agent_dict_duplicates = find_avg_impact(N_with_duplicates)
# simple_synergy_duplicates = find_synergy(simple_avimp_duplicates, simple_agent_dict_duplicates, N_with_duplicates)
simple_avimp_nonadd, simple_avg_size_nonadd, simple_agent_dict_nonadd, N_nonadd = find_size_avg_impact_nonadd_v3(N_)
simple_synergy_nonadd = find_synergy(simple_avimp_nonadd, simple_agent_dict_nonadd, N_nonadd)
simple_avimp_nonadd_duplicates, simple_avg_size_nonadd_duplicates, simple_agent_dict_nonadd_duplicates, N_nonadd_duplicates = find_size_avg_impact_nonadd_v3(N_with_duplicates)
simple_synergy_nonadd_duplicates = find_synergy(simple_avimp_nonadd_duplicates, simple_agent_dict_nonadd_duplicates, N_nonadd_duplicates)

pd.DataFrame(N_bool.sum(axis=1), columns=['size']).to_csv(results_data_path + 'coalition_size_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(N_bool_duplicates.sum(axis=1), columns=['size']).to_csv(results_data_path + 'coalition_size_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
# pd.DataFrame(simple_synergy, columns=['synergy']).to_csv(results_data_path + 'simple_' + domain_name + '.tsv', sep='\t', index=False)
# pd.DataFrame(simple_avimp, columns=['avimp']).to_csv(results_data_path + 'simple_' + domain_name + '_avimp.tsv', sep='\t', index=False)
# pd.DataFrame(simple_synergy_duplicates, columns=['synergy']).to_csv(results_data_path + 'simple_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
# pd.DataFrame(simple_avimp_duplicates, columns=['avimp']).to_csv(results_data_path + 'simple_' + domain_name + '_duplicates_avimp.tsv', sep='\t', index=False)
pd.DataFrame(simple_synergy_nonadd, columns=['synergy']).to_csv(results_data_path + 'simple_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(simple_avimp_nonadd, columns=['avimp']).to_csv(results_data_path + 'simple_' + domain_name + '_avimp.tsv', sep='\t', index=False)
pd.DataFrame(simple_synergy_nonadd_duplicates, columns=['synergy']).to_csv(results_data_path + 'simple_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(simple_avimp_nonadd_duplicates, columns=['avimp']).to_csv(results_data_path + 'simple_' + domain_name + '_duplicates_avimp.tsv', sep='\t', index=False)
pd.DataFrame(har_syn, columns=['synergy']).to_csv(results_data_path + 'harsanyi_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(har_syn_duplicates, columns=['synergy']).to_csv(results_data_path + 'harsanyi_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(gra_syn, columns=['synergy']).to_csv(results_data_path + 'grabisch_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(gra_syn_duplicates, columns=['synergy']).to_csv(results_data_path + 'grabisch_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(np.vstack([synergy_values, coalition_values]).transpose(), columns=['synergy', 'colval']).to_csv(
    results_data_path + 'synergy_index_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(agents_avg_shap, columns=['avshap']).to_csv(results_data_path + 'synergy_index_' + domain_name + '_avshap.tsv', sep='\t', index=False)
pd.DataFrame(np.vstack([synergy_values_duplicates, coalition_values_duplicates]).transpose(), columns=['synergy', 'colval']).to_csv(
    results_data_path + 'synergy_index_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(agents_avg_shap_duplicates, columns=['avshap']).to_csv(results_data_path + 'synergy_index_' + domain_name + '_duplicates_avshap.tsv', sep='\t', index=False)


##################################
## Load real dataset here: Food ##
##################################
domain_id = 1
domain_name = 'food'
print(domain_name)
all_recipes = pd.read_csv("/Users/huhwhat/Documents/synergy-indices/Datasets/yummly/rating_flavor_ingredient_yummly_tab.csv",
                          sep="\t", encoding="utf-8")
obsv = np.array([np.char.split(all_recipes['ingredients'].values.astype('unicode'), ',').tolist(),
                 all_recipes['rating'].values, all_recipes['bitter'].values, all_recipes['meaty'].values,
                 all_recipes['piquant'].values, all_recipes['salty'].values, all_recipes['sour'].values,
                 all_recipes['sweet'].values]).transpose()
obsv_set = obsv.copy()
all_rcp = obsv[:, 0]
ingr = np.unique(np.hstack(all_rcp))
ingr_count = stats.itemfreq(np.hstack(all_rcp))
ingr_count = ingr_count[np.argsort(ingr_count[:, 1].astype(int))][::-1]
top_n = 300
top_n_ingr = ingr_count[:top_n, 0]
select_dict = dict(np.vstack([top_n_ingr.astype(object), np.arange(len(top_n_ingr))]).transpose())
select_rcp = np.array([np.all(np.in1d(x, top_n_ingr)) for x in all_rcp])
print("Creating subset consisting of recipes which can contain only the top " + str(top_n) + " ingredients.")
yummly_N = np.array(obsv[select_rcp])
for i in range(len(yummly_N)):
    yummly_N[i, 0] = set(yummly_N[i, 0])
print("Set creation completed.")

all_agents = np.hstack(np.array([list(x)[:] for x in yummly_N[:, 0]]))
agent_appearances = stats.itemfreq(all_agents).astype(object)
agent_appearances[:, 1] = agent_appearances[:, 1].astype(int)
agent_appearances = agent_appearances[np.argsort(agent_appearances[:, 1].astype(float))[::-1]]
print(len(all_agents), len(yummly_N))
print(len(agent_appearances[agent_appearances[:, 1] > 1]))
print(len(agent_appearances[agent_appearances[:, 1] > 2]))
print(len(agent_appearances[agent_appearances[:, 1] > 3]))
print(agent_appearances[agent_appearances[:, 1] >= 100])

N_ = yummly_N[:, :2]  # Use only ratings as scores
N_[:, 1] -= 2.5  # normalization/rescaling since this is a non-additive domain
N_with_duplicates = N_.copy()
N_bool_duplicates, N_scores_duplicates, all_agents_dict_duplicates = gg.make_bool_N(N_with_duplicates)
N_ = gg.duplicates_to_mean(N_)
N_bool, N_scores, all_agents_dict = gg.make_bool_N(N_)

n_coalitions.append([domain_name, len(N_bool), len(N_bool_duplicates)])
synergy_values, coalition_values, agents_avg_shap = synergy_index(N_bool, N_bool, N_scores)
start = time.time()
synergy_values_duplicates, coalition_values_duplicates, agents_avg_shap_duplicates = synergy_index(N_bool_duplicates, N_bool, N_scores)
stop = time.time()
runtimes_file.write(','.join([domain_name, 'SI', str(stop-start)] + [str(x) for x in N_bool.shape]) + '\n')
start = time.time()
har_syn = har.find_synergy(N_bool, N_bool, N_scores)
har_syn_duplicates = har.find_synergy(N_bool_duplicates, N_bool, N_scores)
stop = time.time()
runtimes_file.write(','.join([domain_name, 'HA', str(stop-start)] + [str(x) for x in N_bool.shape]) + '\n')
start = time.time()
gra_syn = gra.find_synergy(N_bool, N_bool, N_scores)
gra_syn_duplicates = gra.find_synergy(N_bool_duplicates, N_bool, N_scores)
stop = time.time()
runtimes_file.write(','.join([domain_name, 'GR', str(stop-start)] + [str(x) for x in N_bool.shape]) + '\n')
# simple_avimp, simple_agent_dict = find_avg_impact(N_)
# simple_synergy = find_synergy(simple_avimp, simple_agent_dict, N_)
# simple_avimp_duplicates, simple_agent_dict_duplicates = find_avg_impact(N_with_duplicates)
# simple_synergy_duplicates = find_synergy(simple_avimp_duplicates, simple_agent_dict_duplicates, N_with_duplicates)
simple_avimp_nonadd, simple_avg_size_nonadd, simple_agent_dict_nonadd, N_nonadd = find_size_avg_impact_nonadd_v3(N_)
simple_synergy_nonadd = find_synergy(simple_avimp_nonadd, simple_agent_dict_nonadd, N_nonadd)
simple_avimp_nonadd_duplicates, simple_avg_size_nonadd_duplicates, simple_agent_dict_nonadd_duplicates, N_nonadd_duplicates = find_size_avg_impact_nonadd_v3(N_with_duplicates)
simple_synergy_nonadd_duplicates = find_synergy(simple_avimp_nonadd_duplicates, simple_agent_dict_nonadd_duplicates, N_nonadd_duplicates)

pd.DataFrame(N_bool.sum(axis=1), columns=['size']).to_csv(results_data_path + 'coalition_size_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(N_bool_duplicates.sum(axis=1), columns=['size']).to_csv(results_data_path + 'coalition_size_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
# pd.DataFrame(simple_synergy, columns=['synergy']).to_csv(results_data_path + 'simple_' + domain_name + '.tsv', sep='\t', index=False)
# pd.DataFrame(simple_avimp, columns=['avimp']).to_csv(results_data_path + 'simple_' + domain_name + '_avimp.tsv', sep='\t', index=False)
# pd.DataFrame(simple_synergy_duplicates, columns=['synergy']).to_csv(results_data_path + 'simple_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
# pd.DataFrame(simple_avimp_duplicates, columns=['avimp']).to_csv(results_data_path + 'simple_' + domain_name + '_duplicates_avimp.tsv', sep='\t', index=False)
pd.DataFrame(simple_synergy_nonadd, columns=['synergy']).to_csv(results_data_path + 'simple_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(simple_avimp_nonadd, columns=['avimp']).to_csv(results_data_path + 'simple_' + domain_name + '_avimp.tsv', sep='\t', index=False)
pd.DataFrame(simple_synergy_nonadd_duplicates, columns=['synergy']).to_csv(results_data_path + 'simple_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(simple_avimp_nonadd_duplicates, columns=['avimp']).to_csv(results_data_path + 'simple_' + domain_name + '_duplicates_avimp.tsv', sep='\t', index=False)
pd.DataFrame(har_syn, columns=['synergy']).to_csv(results_data_path + 'harsanyi_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(har_syn_duplicates, columns=['synergy']).to_csv(results_data_path + 'harsanyi_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(gra_syn, columns=['synergy']).to_csv(results_data_path + 'grabisch_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(gra_syn_duplicates, columns=['synergy']).to_csv(results_data_path + 'grabisch_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(np.vstack([synergy_values, coalition_values]).transpose(), columns=['synergy', 'colval']).to_csv(
    results_data_path + 'synergy_index_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(agents_avg_shap, columns=['avshap']).to_csv(results_data_path + 'synergy_index_' + domain_name + '_avshap.tsv', sep='\t', index=False)
pd.DataFrame(np.vstack([synergy_values_duplicates, coalition_values_duplicates]).transpose(), columns=['synergy', 'colval']).to_csv(
    results_data_path + 'synergy_index_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(agents_avg_shap_duplicates, columns=['avshap']).to_csv(results_data_path + 'synergy_index_' + domain_name + '_duplicates_avshap.tsv', sep='\t', index=False)


######################################
## Load real dataset here: Perfumes ##
######################################
domain_id = 2
domain_name = 'perfumes'
print(domain_name)

fragrantica_df = pd.DataFrame.from_csv('/Users/huhwhat/Documents/synergy-indices/Datasets/fragrantica_df.tsv', index_col=None, sep='\t')
print(str(len(np.unique(re.split(';', ';'.join(fragrantica_df.all_notes[fragrantica_df.all_notes.notnull()].values))))) + ' unique notes') ## 1168 unique notes

N_df = fragrantica_df[['all_notes', 'rating']][np.all([fragrantica_df.all_notes.notnull(), (fragrantica_df.n_ratings > 10)], axis=0)]
N_ = np.array([[set(re.split(';', obsv[0])), obsv[1]] for obsv in N_df.values]).astype(object)

import scipy.stats as stats
all_agents = np.hstack(np.array([list(x)[:] for x in N_[:, 0]]))
agent_appearances = stats.itemfreq(all_agents).astype(object)
agent_appearances[:, 1] = agent_appearances[:, 1].astype(int)
agent_appearances = agent_appearances[np.argsort(agent_appearances[:, 1].astype(float))[::-1]]
print(len(agent_appearances[agent_appearances[:, 1] > 1]))
print(len(agent_appearances[agent_appearances[:, 1] > 2]))
print(len(agent_appearances[agent_appearances[:, 1] > 3]))

## Restrict data to only perfumes using more popular ingredients
print(len(agent_appearances[agent_appearances[:, 1] >= 10]))  # 490
print(len(agent_appearances[agent_appearances[:, 1] >= 20]))  # 365
select_agents = set(agent_appearances[agent_appearances[:, 1] >= 20][:, 0])
N_ = np.array([[obsv[0], obsv[1]] for obsv in N_ if select_agents.issuperset(obsv[0])])
N_[:, 1] -= 2.5  # normalization/rescaling since this is a non-additive domain (note that the lowest score in perfumes is 1.54, so maybe subtract by mean instead)
N_with_duplicates = N_.copy()
N_bool_duplicates, N_scores_duplicates, all_agents_dict_duplicates = gg.make_bool_N(N_with_duplicates)
N_ = gg.duplicates_to_mean(N_)
N_bool, N_scores, all_agents_dict = gg.make_bool_N(N_)

n_coalitions.append([domain_name, len(N_bool), len(N_bool_duplicates)])
synergy_values, coalition_values, agents_avg_shap = synergy_index(N_bool, N_bool, N_scores)
start = time.time()
synergy_values_duplicates, coalition_values_duplicates, agents_avg_shap_duplicates = synergy_index(N_bool_duplicates, N_bool, N_scores)
stop = time.time()
runtimes_file.write(','.join([domain_name, 'SI', str(stop-start)] + [str(x) for x in N_bool.shape]) + '\n')
start = time.time()
har_syn = har.find_synergy(N_bool, N_bool, N_scores)
har_syn_duplicates = har.find_synergy(N_bool_duplicates, N_bool, N_scores)
stop = time.time()
runtimes_file.write(','.join([domain_name, 'HA', str(stop-start)] + [str(x) for x in N_bool.shape]) + '\n')
start = time.time()
gra_syn = gra.find_synergy(N_bool, N_bool, N_scores)
gra_syn_duplicates = gra.find_synergy(N_bool_duplicates, N_bool, N_scores)
stop = time.time()
runtimes_file.write(','.join([domain_name, 'GR', str(stop-start)] + [str(x) for x in N_bool.shape]) + '\n')
# simple_avimp, simple_agent_dict = find_avg_impact(N_)
# simple_synergy = find_synergy(simple_avimp, simple_agent_dict, N_)
# simple_avimp_duplicates, simple_agent_dict_duplicates = find_avg_impact(N_with_duplicates)
# simple_synergy_duplicates = find_synergy(simple_avimp_duplicates, simple_agent_dict_duplicates, N_with_duplicates)
simple_avimp_nonadd, simple_avg_size_nonadd, simple_agent_dict_nonadd, N_nonadd = find_size_avg_impact_nonadd_v3(N_)
simple_synergy_nonadd = find_synergy(simple_avimp_nonadd, simple_agent_dict_nonadd, N_nonadd)
simple_avimp_nonadd_duplicates, simple_avg_size_nonadd_duplicates, simple_agent_dict_nonadd_duplicates, N_nonadd_duplicates = find_size_avg_impact_nonadd_v3(N_with_duplicates)
simple_synergy_nonadd_duplicates = find_synergy(simple_avimp_nonadd_duplicates, simple_agent_dict_nonadd_duplicates, N_nonadd_duplicates)

pd.DataFrame(N_bool.sum(axis=1), columns=['size']).to_csv(results_data_path + 'coalition_size_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(N_bool_duplicates.sum(axis=1), columns=['size']).to_csv(results_data_path + 'coalition_size_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
# pd.DataFrame(simple_synergy, columns=['synergy']).to_csv(results_data_path + 'simple_' + domain_name + '.tsv', sep='\t', index=False)
# pd.DataFrame(simple_avimp, columns=['avimp']).to_csv(results_data_path + 'simple_' + domain_name + '_avimp.tsv', sep='\t', index=False)
# pd.DataFrame(simple_synergy_duplicates, columns=['synergy']).to_csv(results_data_path + 'simple_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
# pd.DataFrame(simple_avimp_duplicates, columns=['avimp']).to_csv(results_data_path + 'simple_' + domain_name + '_duplicates_avimp.tsv', sep='\t', index=False)
pd.DataFrame(simple_synergy_nonadd, columns=['synergy']).to_csv(results_data_path + 'simple_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(simple_avimp_nonadd, columns=['avimp']).to_csv(results_data_path + 'simple_' + domain_name + '_avimp.tsv', sep='\t', index=False)
pd.DataFrame(simple_synergy_nonadd_duplicates, columns=['synergy']).to_csv(results_data_path + 'simple_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(simple_avimp_nonadd_duplicates, columns=['avimp']).to_csv(results_data_path + 'simple_' + domain_name + '_duplicates_avimp.tsv', sep='\t', index=False)
pd.DataFrame(har_syn, columns=['synergy']).to_csv(results_data_path + 'harsanyi_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(har_syn_duplicates, columns=['synergy']).to_csv(results_data_path + 'harsanyi_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(gra_syn, columns=['synergy']).to_csv(results_data_path + 'grabisch_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(gra_syn_duplicates, columns=['synergy']).to_csv(results_data_path + 'grabisch_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(np.vstack([synergy_values, coalition_values]).transpose(), columns=['synergy', 'colval']).to_csv(
    results_data_path + 'synergy_index_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(agents_avg_shap, columns=['avshap']).to_csv(results_data_path + 'synergy_index_' + domain_name + '_avshap.tsv', sep='\t', index=False)
pd.DataFrame(np.vstack([synergy_values_duplicates, coalition_values_duplicates]).transpose(), columns=['synergy', 'colval']).to_csv(
    results_data_path + 'synergy_index_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(agents_avg_shap_duplicates, columns=['avshap']).to_csv(results_data_path + 'synergy_index_' + domain_name + '_duplicates_avshap.tsv', sep='\t', index=False)


##########################################
## Load real dataset here Movie, scores ##
##########################################
domain_id = 3
domain_name = 'movies_score'
print(domain_name)


movielens_movies_file = "/Users/huhwhat/Documents/synergy-indices/Datasets/movies/movies.dat"
movielens_directors_file = "/Users/huhwhat/Documents/synergy-indices/Datasets/movies/movie_directors.dat"
movielens_actors_file = "/Users/huhwhat/Documents/synergy-indices/Datasets/movies/movie_actors.dat"
movielens_ratings_file = "/Users/huhwhat/Documents/synergy-indices/Datasets/movies/user_ratedmovies.dat"
usa_gross_file = "/Users/huhwhat/Documents/synergy-indices/Datasets/movies/usa_gross.tsv"

movielens_movies = pd.read_csv(movielens_movies_file, sep="\t", header=0)
movielens_directors = pd.read_csv(movielens_directors_file, sep="\t", header=0)
movielens_actors = pd.read_csv(movielens_actors_file, sep="\t", header=0)
movielens_ratings = pd.read_csv(movielens_ratings_file, sep="\t", header=0)
usa_gross = pd.read_csv(usa_gross_file, delimiter="\t")

id_movies_directors = np.intersect1d(movielens_movies["id"].values, movielens_directors["movieID"].unique())
id_movies_directors_actors = np.intersect1d(id_movies_directors, movielens_actors["movieID"].unique())

# movies_N_df_columns = ["movieID", "director", "actors", "director_actors", "rtAllCriticsRating", "rtAllCriticsNumFresh",
#                        "rtAllCriticsNumRotten", "rtAllCriticsScore", "rtTopCriticsRating", "rtTopCriticsNumFresh",
#                        "rtTopCriticsNumRotten", "rtTopCriticsScore", "rtAudienceRating", "rtAudienceScore"]
# metric_columns = ["rtAllCriticsRating", "rtAllCriticsNumFresh", "rtAllCriticsNumRotten", "rtAllCriticsScore",
#                   "rtTopCriticsRating", "rtTopCriticsNumFresh", "rtTopCriticsNumRotten", "rtTopCriticsScore",
#                   "rtAudienceRating", "rtAudienceScore"]
movies_N_df_columns = ["movieID", "director", "actors", "director_actors", "rtAllCriticsRating", "rtAllCriticsScore",
                       "rtTopCriticsRating", "rtTopCriticsScore", "rtAudienceRating", "rtAudienceScore", "usa_gross"]
metric_columns = ["rtAllCriticsRating", "rtAllCriticsScore", "rtTopCriticsRating", "rtTopCriticsScore",
                  "rtAudienceRating", "rtAudienceScore"]

movies_N_df = pd.DataFrame(columns=movies_N_df_columns)
for id_ in id_movies_directors_actors:
    movie_stats = movielens_movies[metric_columns][movielens_movies["id"] == id_].values[0]
    directors = set(movielens_directors["directorID"][movielens_directors["movieID"] == id_].values)
    actors = set(movielens_actors["actorID"][movielens_actors["movieID"] == id_].values)
    director_actors = actors.copy()
    if len(usa_gross[usa_gross["id"] == id_].values) > 0:
        gross = usa_gross[usa_gross["id"] == id_]["max_gross"].values[0]
    else:
        gross = np.nan
    for d_ in directors:
        director_actors.add(d_)
    row_to_append = dict(zip(movies_N_df_columns, np.hstack([[id_, directors, actors, director_actors], movie_stats, [gross]])))
    movies_N_df = movies_N_df.append(row_to_append, ignore_index=True)

movies_N_ = movies_N_df[['director_actors', 'rtAllCriticsRating']].values
movies_N_ = movies_N_[movies_N_[:, 1] != '\\N']
movies_N_ = movies_N_[movies_N_[:, 1] != '0']
movies_N_ = movies_N_[np.isfinite(movies_N_[:, 1].astype(float))]
movies_N_[:, 1] = movies_N_[:, 1].astype(float)
all_actors = np.unique(np.hstack([list(group_) for group_ in movies_N_[:, 0]]))
actors_appearances = stats.itemfreq(np.hstack([list(group_) for group_ in movies_N_[:, 0]])).astype(object)
actors_appearances[:, 1] = actors_appearances[:, 1].astype(int)
select_actors = set(actors_appearances[(actors_appearances[:, 1] >= 3), 0])
N_ = []
for group_, score_ in movies_N_:
    if group_ is not None:
        if group_.issubset(select_actors):
            N_.append([group_, score_])
N_ = np.array(N_).astype(object)
N_[:, 1] = N_[:, 1].astype(float)
for i in range(len(N_)):
    if N_[i, 0] is not None:
        if N_[i, 0].__contains__(""):
            N_[i, 0].remove("")
print('# movies: ', len(N_))
print('# actors: ', len(np.unique(np.hstack([list(group_) for group_ in N_[:, 0]]))))
N_[:, 1] -= 5.  # normalization/rescaling since this is a non-additive domain (note that the lowest score in perfumes is 1.7, so maybe subtract by mean instead)
N_with_duplicates = N_.copy()
N_bool_duplicates, N_scores_duplicates, all_agents_dict_duplicates = gg.make_bool_N(N_with_duplicates)
N_ = gg.duplicates_to_mean(N_)
N_bool, N_scores, all_agents_dict = gg.make_bool_N(N_)
save_agent_dict(all_agents_dict, results_data_path + 'agents_dict_' + domain_name + '.tsv')

n_coalitions.append([domain_name, len(N_bool), len(N_bool_duplicates)])
synergy_values, coalition_values, agents_avg_shap = synergy_index(N_bool, N_bool, N_scores)
start = time.time()
synergy_values_duplicates, coalition_values_duplicates, agents_avg_shap_duplicates = synergy_index(N_bool_duplicates, N_bool, N_scores)
stop = time.time()
runtimes_file.write(','.join([domain_name, 'SI', str(stop-start)] + [str(x) for x in N_bool.shape]) + '\n')
start = time.time()
har_syn = har.find_synergy(N_bool, N_bool, N_scores)
har_syn_duplicates = har.find_synergy(N_bool_duplicates, N_bool, N_scores)
stop = time.time()
runtimes_file.write(','.join([domain_name, 'HA', str(stop-start)] + [str(x) for x in N_bool.shape]) + '\n')
start = time.time()
gra_syn = gra.find_synergy(N_bool, N_bool, N_scores)
gra_syn_duplicates = gra.find_synergy(N_bool_duplicates, N_bool, N_scores)
stop = time.time()
runtimes_file.write(','.join([domain_name, 'GR', str(stop-start)] + [str(x) for x in N_bool.shape]) + '\n')
# simple_avimp, simple_agent_dict = find_avg_impact(N_)
# simple_synergy = find_synergy(simple_avimp, simple_agent_dict, N_)
# simple_avimp_duplicates, simple_agent_dict_duplicates = find_avg_impact(N_with_duplicates)
# simple_synergy_duplicates = find_synergy(simple_avimp_duplicates, simple_agent_dict_duplicates, N_with_duplicates)
simple_avimp_nonadd, simple_avg_size_nonadd, simple_agent_dict_nonadd, N_nonadd = find_size_avg_impact_nonadd_v3(N_)
simple_synergy_nonadd = find_synergy(simple_avimp_nonadd, simple_agent_dict_nonadd, N_nonadd)
simple_avimp_nonadd_duplicates, simple_avg_size_nonadd_duplicates, simple_agent_dict_nonadd_duplicates, N_nonadd_duplicates = find_size_avg_impact_nonadd_v3(N_with_duplicates)
simple_synergy_nonadd_duplicates = find_synergy(simple_avimp_nonadd_duplicates, simple_agent_dict_nonadd_duplicates, N_nonadd_duplicates)

pd.DataFrame(N_bool.sum(axis=1), columns=['size']).to_csv(results_data_path + 'coalition_size_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(N_bool_duplicates.sum(axis=1), columns=['size']).to_csv(results_data_path + 'coalition_size_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
# pd.DataFrame(simple_synergy, columns=['synergy']).to_csv(results_data_path + 'simple_' + domain_name + '.tsv', sep='\t', index=False)
# pd.DataFrame(simple_avimp, columns=['avimp']).to_csv(results_data_path + 'simple_' + domain_name + '_avimp.tsv', sep='\t', index=False)
# pd.DataFrame(simple_synergy_duplicates, columns=['synergy']).to_csv(results_data_path + 'simple_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
# pd.DataFrame(simple_avimp_duplicates, columns=['avimp']).to_csv(results_data_path + 'simple_' + domain_name + '_duplicates_avimp.tsv', sep='\t', index=False)
pd.DataFrame(simple_synergy_nonadd, columns=['synergy']).to_csv(results_data_path + 'simple_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(simple_avimp_nonadd, columns=['avimp']).to_csv(results_data_path + 'simple_' + domain_name + '_avimp.tsv', sep='\t', index=False)
pd.DataFrame(simple_synergy_nonadd_duplicates, columns=['synergy']).to_csv(results_data_path + 'simple_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(simple_avimp_nonadd_duplicates, columns=['avimp']).to_csv(results_data_path + 'simple_' + domain_name + '_duplicates_avimp.tsv', sep='\t', index=False)
pd.DataFrame(har_syn, columns=['synergy']).to_csv(results_data_path + 'harsanyi_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(har_syn_duplicates, columns=['synergy']).to_csv(results_data_path + 'harsanyi_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(gra_syn, columns=['synergy']).to_csv(results_data_path + 'grabisch_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(gra_syn_duplicates, columns=['synergy']).to_csv(results_data_path + 'grabisch_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(np.vstack([synergy_values, coalition_values]).transpose(), columns=['synergy', 'colval']).to_csv(
    results_data_path + 'synergy_index_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(agents_avg_shap, columns=['avshap']).to_csv(results_data_path + 'synergy_index_' + domain_name + '_avshap.tsv', sep='\t', index=False)
pd.DataFrame(np.vstack([synergy_values_duplicates, coalition_values_duplicates]).transpose(), columns=['synergy', 'colval']).to_csv(
    results_data_path + 'synergy_index_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(agents_avg_shap_duplicates, columns=['avshap']).to_csv(results_data_path + 'synergy_index_' + domain_name + '_duplicates_avshap.tsv', sep='\t', index=False)


#########################################
## Load real dataset here Movie, gross ##
#########################################
domain_id = 4
domain_name = 'movies_gross'
print(domain_name)

movies_N_ = movies_N_df[['director_actors', 'usa_gross']].values
movies_N_ = movies_N_[movies_N_[:, 1] != '\\N']
movies_N_ = movies_N_[movies_N_[:, 1] != '0']
movies_N_ = movies_N_[np.isfinite(movies_N_[:, 1].astype(float))]
movies_N_[:, 1] = movies_N_[:, 1].astype(float)
all_actors = np.unique(np.hstack([list(group_) for group_ in movies_N_[:, 0]]))
actors_appearances = stats.itemfreq(np.hstack([list(group_) for group_ in movies_N_[:, 0]])).astype(object)
actors_appearances[:, 1] = actors_appearances[:, 1].astype(int)
select_actors = set(actors_appearances[(actors_appearances[:, 1] >= 3), 0])
N_ = []
for group_, score_ in movies_N_:
    if group_ is not None:
        if group_.issubset(select_actors):
            N_.append([group_, score_])
N_ = np.array(N_).astype(object)
N_[:, 1] = N_[:, 1].astype(float)
for i in range(len(N_)):
    if N_[i, 0] is not None:
        if N_[i, 0].__contains__(""):
            N_[i, 0].remove("")
print('# movies: ', len(N_))
print('# actors: ', len(np.unique(np.hstack([list(group_) for group_ in N_[:, 0]]))))
N_with_duplicates = N_.copy()
N_bool_duplicates, N_scores_duplicates, all_agents_dict_duplicates = gg.make_bool_N(N_with_duplicates)
N_ = gg.duplicates_to_mean(N_)
N_bool, N_scores, all_agents_dict = gg.make_bool_N(N_)
save_agent_dict(all_agents_dict, results_data_path + 'agents_dict_' + domain_name + '.tsv')

n_coalitions.append([domain_name, len(N_bool), len(N_bool_duplicates)])
synergy_values, coalition_values, agents_avg_shap = synergy_index(N_bool, N_bool, N_scores)
start = time.time()
synergy_values_duplicates, coalition_values_duplicates, agents_avg_shap_duplicates = synergy_index(N_bool_duplicates, N_bool, N_scores)
stop = time.time()
runtimes_file.write(','.join([domain_name, 'SI', str(stop-start)] + [str(x) for x in N_bool.shape]) + '\n')
start = time.time()
har_syn = har.find_synergy(N_bool, N_bool, N_scores)
har_syn_duplicates = har.find_synergy(N_bool_duplicates, N_bool, N_scores)
stop = time.time()
runtimes_file.write(','.join([domain_name, 'HA', str(stop-start)] + [str(x) for x in N_bool.shape]) + '\n')
start = time.time()
gra_syn = gra.find_synergy(N_bool, N_bool, N_scores)
gra_syn_duplicates = gra.find_synergy(N_bool_duplicates, N_bool, N_scores)
stop = time.time()
runtimes_file.write(','.join([domain_name, 'GR', str(stop-start)] + [str(x) for x in N_bool.shape]) + '\n')
simple_avimp, simple_agent_dict = find_avg_impact(N_)
simple_synergy = find_synergy(simple_avimp, simple_agent_dict, N_)
simple_avimp_duplicates, simple_agent_dict_duplicates = find_avg_impact(N_with_duplicates)
simple_synergy_duplicates = find_synergy(simple_avimp_duplicates, simple_agent_dict_duplicates, N_with_duplicates)

pd.DataFrame(N_bool.sum(axis=1), columns=['size']).to_csv(results_data_path + 'coalition_size_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(N_bool_duplicates.sum(axis=1), columns=['size']).to_csv(results_data_path + 'coalition_size_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(simple_synergy, columns=['synergy']).to_csv(results_data_path + 'simple_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(simple_avimp, columns=['avimp']).to_csv(results_data_path + 'simple_' + domain_name + '_avimp.tsv', sep='\t', index=False)
pd.DataFrame(simple_synergy_duplicates, columns=['synergy']).to_csv(results_data_path + 'simple_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(simple_avimp_duplicates, columns=['avimp']).to_csv(results_data_path + 'simple_' + domain_name + '_duplicates_avimp.tsv', sep='\t', index=False)
pd.DataFrame(har_syn, columns=['synergy']).to_csv(results_data_path + 'harsanyi_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(har_syn_duplicates, columns=['synergy']).to_csv(results_data_path + 'harsanyi_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(gra_syn, columns=['synergy']).to_csv(results_data_path + 'grabisch_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(gra_syn_duplicates, columns=['synergy']).to_csv(results_data_path + 'grabisch_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(np.vstack([synergy_values, coalition_values]).transpose(), columns=['synergy', 'colval']).to_csv(
    results_data_path + 'synergy_index_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(agents_avg_shap, columns=['avshap']).to_csv(results_data_path + 'synergy_index_' + domain_name + '_avshap.tsv', sep='\t', index=False)
pd.DataFrame(np.vstack([synergy_values_duplicates, coalition_values_duplicates]).transpose(), columns=['synergy', 'colval']).to_csv(
    results_data_path + 'synergy_index_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(agents_avg_shap_duplicates, columns=['avshap']).to_csv(results_data_path + 'synergy_index_' + domain_name + '_duplicates_avshap.tsv', sep='\t', index=False)


####################################
## Load real dataset here: Papers ##
####################################
domain_id = 5
domain_name = 'papers'
print(domain_name)
# import sqlite3 as lite
# import scipy.stats as stats
# db_path = "/Users/huhwhat/Documents/arnet_miner_v6.db"
# con = lite.connect(db_path)
# cur = con.cursor()
# print(np.array(cur.execute('PRAGMA table_info(Papers);').fetchall()))
# print(np.array(cur.execute('SELECT * FROM Papers LIMIT 10;').fetchall()))
# # all_authors_citations = np.array(cur.execute('SELECT Authors, Citations FROM Papers').fetchall())
# all_authors_year_venue_citations = np.array(cur.execute('SELECT Authors, Citations, Year, Venue FROM Papers').fetchall())
# all_authors_year_venue_citations = all_authors_year_venue_citations[all_authors_year_venue_citations[:, 1] != -1]
# all_authors_year_venue_citations = all_authors_year_venue_citations[all_authors_year_venue_citations[:, 0] != u""]
# all_authors = all_authors_year_venue_citations[:, 0]
# all_authors = np.array([re.split(',', x) for x in all_authors if x is not None])
#
# import bs4
# from bs4 import BeautifulSoup
# filepath_ = '/Users/huhwhat/Documents/arnetminer/The h Index for Computer Science (modified).html'  # Modification consists of replacing <!a href> tags with <a href>
# html_doc = open(filepath_).read()
# filesoup = BeautifulSoup(html_doc)
# authors_h_loc = []
# institution_set = set(re.findall('\((.*?)\)', html_doc))
# institution_set.remove('Academia Europaea')
# for result_ in filesoup.find_all('a'):
#     if result_.text != '' and type(result_.next_sibling) == bs4.element.NavigableString:
#         if len(re.findall('\((.*?)\)', result_.next_sibling)) != 0:
#             if set(re.findall('\((.*?)\)', result_.next_sibling)).issubset(institution_set):
#                 institution = re.findall('\((.*?)\)', result_.next_sibling)[0]
#                 author = result_.text
#                 if type(result_.previous_sibling) != bs4.element.Tag and result_.previous_sibling != '\n':
#                     h_idx = int(re.split('\n', result_.previous_sibling)[1])
#                 else:
#                     h_idx_str = re.split('\n', result_.previous_sibling.previous_sibling.text)
#                     if len(h_idx_str) > 1:
#                         h_idx = int(h_idx_str[1])
#                     else:
#                         h_idx = int(re.split('\n', result_.previous_sibling.previous_sibling.previous_sibling.text)[1])
#                 authors_h_loc.append([author, h_idx, institution])
#
# authors_h_loc = pd.DataFrame(data=authors_h_loc, columns=['author', 'h_index', 'institution'])
# select_authors_palsberg = set(authors_h_loc['author'].values)
# papers_N = []
# for group_, score_, year_, venue_ in all_authors_year_venue_citations:
#     if group_ is not None:
#         group_members = re.split(',', group_)
#         if set(group_members).issubset(select_authors_palsberg):
#            papers_N.append([group_members, score_, year_, venue_])
# papers_N = np.array(papers_N)
# papers_N[:, 0] = [';'.join(x) for x in papers_N[:, 0]]
# pd.DataFrame(papers_N[:, :2], columns = ['authors', 'citations']).to_csv('/Users/huhwhat/Documents/synergy-indices/Datasets/papers/papers_N.csv', index=False, encoding='utf-8')
papers_N = pd.read_csv('/Users/huhwhat/Documents/synergy-indices/Datasets/papers/papers_N.csv')
papers_N = np.array(papers_N)
papers_N[:, 0] = [re.split(';', x) for x in papers_N[:, 0]]
papers_N = papers_N.tolist()
N_ = papers_N
N_ = np.array(N_).astype(object)
N_[:, 1] = N_[:, 1].astype(float)
N_ = N_[N_[:, 1] != -1]
N_ = N_[N_[:, 0] != u""]
for i in range(len(N_)):
    if N_[i, 0] is not None:
        if N_[i, 0].__contains__(""):
            N_[i, 0].remove("")
print('# papers: ', len(N_))
print('# authors: ', len(np.unique(np.hstack(N_[:, 0]))))
N_with_duplicates = N_.copy()
N_bool_duplicates, N_scores_duplicates, all_agents_dict_duplicates = gg.make_bool_N(N_with_duplicates)
N_ = gg.duplicates_to_mean(N_[:, :2])
N_bool, N_scores, all_agents_dict = gg.make_bool_N(N_)

n_coalitions.append([domain_name, len(N_bool), len(N_bool_duplicates)])
synergy_values, coalition_values, agents_avg_shap = synergy_index(N_bool, N_bool, N_scores)
start = time.time()
synergy_values_duplicates, coalition_values_duplicates, agents_avg_shap_duplicates = synergy_index(N_bool_duplicates, N_bool, N_scores)
stop = time.time()
runtimes_file.write(','.join([domain_name, 'SI', str(stop-start)] + [str(x) for x in N_bool.shape]) + '\n')
start = time.time()
har_syn = har.find_synergy(N_bool, N_bool, N_scores)
har_syn_duplicates = har.find_synergy(N_bool_duplicates, N_bool, N_scores)
stop = time.time()
runtimes_file.write(','.join([domain_name, 'HA', str(stop-start)] + [str(x) for x in N_bool.shape]) + '\n')
start = time.time()
gra_syn = gra.find_synergy(N_bool, N_bool, N_scores)
gra_syn_duplicates = gra.find_synergy(N_bool_duplicates, N_bool, N_scores)
stop = time.time()
runtimes_file.write(','.join([domain_name, 'GR', str(stop-start)] + [str(x) for x in N_bool.shape]) + '\n')
simple_avimp, simple_agent_dict = find_avg_impact(N_)
simple_synergy = find_synergy(simple_avimp, simple_agent_dict, N_)
simple_avimp_duplicates, simple_agent_dict_duplicates = find_avg_impact(N_with_duplicates[:, :2])
simple_synergy_duplicates = find_synergy(simple_avimp_duplicates, simple_agent_dict_duplicates, N_with_duplicates[:, :2])

pd.DataFrame(N_bool.sum(axis=1), columns=['size']).to_csv(results_data_path + 'coalition_size_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(N_bool_duplicates.sum(axis=1), columns=['size']).to_csv(results_data_path + 'coalition_size_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(simple_synergy, columns=['synergy']).to_csv(results_data_path + 'simple_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(simple_avimp, columns=['avimp']).to_csv(results_data_path + 'simple_' + domain_name + '_avimp.tsv', sep='\t', index=False)
pd.DataFrame(simple_synergy_duplicates, columns=['synergy']).to_csv(results_data_path + 'simple_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(simple_avimp_duplicates, columns=['avimp']).to_csv(results_data_path + 'simple_' + domain_name + '_duplicates_avimp.tsv', sep='\t', index=False)
pd.DataFrame(har_syn, columns=['synergy']).to_csv(results_data_path + 'harsanyi_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(har_syn_duplicates, columns=['synergy']).to_csv(results_data_path + 'harsanyi_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(gra_syn, columns=['synergy']).to_csv(results_data_path + 'grabisch_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(gra_syn_duplicates, columns=['synergy']).to_csv(results_data_path + 'grabisch_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(np.vstack([synergy_values, coalition_values]).transpose(), columns=['synergy', 'colval']).to_csv(
    results_data_path + 'synergy_index_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(agents_avg_shap, columns=['avshap']).to_csv(results_data_path + 'synergy_index_' + domain_name + '_avshap.tsv', sep='\t', index=False)
pd.DataFrame(np.vstack([synergy_values_duplicates, coalition_values_duplicates]).transpose(), columns=['synergy', 'colval']).to_csv(
    results_data_path + 'synergy_index_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(agents_avg_shap_duplicates, columns=['avshap']).to_csv(results_data_path + 'synergy_index_' + domain_name + '_duplicates_avshap.tsv', sep='\t', index=False)


########################################
## Load real dataset here: Basketball ##
########################################
domain_id = 6
domain_name = 'bball'
print(domain_name)
bball_N_df = pd.io.parsers.read_table('/Users/huhwhat/Documents/synergy-indices/Datasets/basketball/bball_N.tsv',
                                index_col=None, sep='\t')
N_ = np.array([[set(re.split(',', obsv[0])), obsv[1]] for obsv in bball_N_df.values]).astype(object)

all_agents = np.hstack(np.array([list(x)[:] for x in N_[:, 0]]))
agent_appearances = stats.itemfreq(all_agents).astype(object)
agent_appearances[:, 1] = agent_appearances[:, 1].astype(int)
agent_appearances = agent_appearances[np.argsort(agent_appearances[:, 1].astype(float))[::-1]]
print(len(agent_appearances[agent_appearances[:, 1] > 1]))
print(len(agent_appearances[agent_appearances[:, 1] > 2]))
print(len(agent_appearances[agent_appearances[:, 1] > 3]))

top_n = 300
print(len(agent_appearances[agent_appearances[:, 1] > top_n]))
top_n_agents = agent_appearances[:top_n, 0]
print("Creating subset consisting of groups which can contain only the top " + str(top_n) + " agents.")
top_n_agents_set = set(top_n_agents)
select_groups = []
for coalition, score in N_:
    if coalition.issubset(top_n_agents_set):
        select_groups.append([coalition, score])
N_ = np.array(select_groups)
print("Set creation completed.")
N_with_duplicates = N_.copy()
N_bool_duplicates, N_scores_duplicates, all_agents_dict_duplicates = gg.make_bool_N(N_with_duplicates)
N_ = gg.duplicates_to_mean(N_)
N_bool, N_scores, all_agents_dict = gg.make_bool_N(N_)

n_coalitions.append([domain_name, len(N_bool), len(N_bool_duplicates)])
synergy_values, coalition_values, agents_avg_shap = synergy_index(N_bool, N_bool, N_scores)
start = time.time()
synergy_values_duplicates, coalition_values_duplicates, agents_avg_shap_duplicates = synergy_index(N_bool_duplicates, N_bool, N_scores)
stop = time.time()
runtimes_file.write(','.join([domain_name, 'SI', str(stop-start)] + [str(x) for x in N_bool.shape]) + '\n')
start = time.time()
har_syn = har.find_synergy(N_bool, N_bool, N_scores)
har_syn_duplicates = har.find_synergy(N_bool_duplicates, N_bool, N_scores)
stop = time.time()
runtimes_file.write(','.join([domain_name, 'HA', str(stop-start)] + [str(x) for x in N_bool.shape]) + '\n')
start = time.time()
gra_syn = gra.find_synergy(N_bool, N_bool, N_scores)
gra_syn_duplicates = gra.find_synergy(N_bool_duplicates, N_bool, N_scores)
stop = time.time()
runtimes_file.write(','.join([domain_name, 'GR', str(stop-start)] + [str(x) for x in N_bool.shape]) + '\n')
simple_avimp, simple_agent_dict = find_avg_impact(N_)
simple_synergy = find_synergy(simple_avimp, simple_agent_dict, N_)
simple_avimp_duplicates, simple_agent_dict_duplicates = find_avg_impact(N_with_duplicates)
simple_synergy_duplicates = find_synergy(simple_avimp_duplicates, simple_agent_dict_duplicates, N_with_duplicates)

pd.DataFrame(N_bool.sum(axis=1), columns=['size']).to_csv(results_data_path + 'coalition_size_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(N_bool_duplicates.sum(axis=1), columns=['size']).to_csv(results_data_path + 'coalition_size_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(simple_synergy, columns=['synergy']).to_csv(results_data_path + 'simple_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(simple_avimp, columns=['avimp']).to_csv(results_data_path + 'simple_' + domain_name + '_avimp.tsv', sep='\t', index=False)
pd.DataFrame(simple_synergy_duplicates, columns=['synergy']).to_csv(results_data_path + 'simple_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(simple_avimp_duplicates, columns=['avimp']).to_csv(results_data_path + 'simple_' + domain_name + '_duplicates_avimp.tsv', sep='\t', index=False)
pd.DataFrame(har_syn, columns=['synergy']).to_csv(results_data_path + 'harsanyi_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(har_syn_duplicates, columns=['synergy']).to_csv(results_data_path + 'harsanyi_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(gra_syn, columns=['synergy']).to_csv(results_data_path + 'grabisch_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(gra_syn_duplicates, columns=['synergy']).to_csv(results_data_path + 'grabisch_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(np.vstack([synergy_values, coalition_values]).transpose(), columns=['synergy', 'colval']).to_csv(
    results_data_path + 'synergy_index_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(agents_avg_shap, columns=['avshap']).to_csv(results_data_path + 'synergy_index_' + domain_name + '_avshap.tsv', sep='\t', index=False)
pd.DataFrame(np.vstack([synergy_values_duplicates, coalition_values_duplicates]).transpose(), columns=['synergy', 'colval']).to_csv(
    results_data_path + 'synergy_index_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(agents_avg_shap_duplicates, columns=['avshap']).to_csv(results_data_path + 'synergy_index_' + domain_name + '_duplicates_avshap.tsv', sep='\t', index=False)


########################################
## Create and use random dataset here ##
########################################
domain_id = 7
domain_name = 'random'
print(domain_name)
n_coalitions = 50000
n_agents = 3000


def make_random_coalitions(n_coalitions, n_agents, size_dist='uniform', score_dist='uniform'):
    agent_space = np.arange(n_agents).astype(str)
    if size_dist == 'uniform':
        coalition_sizes = np.random.uniform(1, n_agents, n_coalitions).round().astype(int)
    if score_dist == 'uniform':
        scores = np.random.uniform(1, 100, n_coalitions).round().astype(int)
    coalitions = [set(np.random.choice(agent_space, size=x, replace=False)) for x in coalition_sizes]
    return np.array(zip(coalitions, scores))

N_ = make_random_coalitions(n_coalitions, n_agents)
N_[:, 1] -= 50
N_with_duplicates = N_.copy()
N_bool_duplicates, N_scores_duplicates, all_agents_dict_duplicates = gg.make_bool_N(N_with_duplicates)
N_ = gg.duplicates_to_mean(N_[:, :2])
N_bool, N_scores, all_agents_dict = gg.make_bool_N(N_)

start = time.time()
synergy_values_duplicates, coalition_values_duplicates, agents_avg_shap_duplicates = synergy_index(N_bool_duplicates, N_bool, N_scores)
stop = time.time()
runtimes_file.write(','.join([domain_name, 'SI', str(stop-start)] + [str(x) for x in N_bool.shape]) + '\n')
# print(','.join([domain_name, 'SI', str(stop-start)] + [str(x) for x in N_bool.shape]))
start = time.time()
har_syn = har.find_synergy(N_bool, N_bool, N_scores)
har_syn_duplicates = har.find_synergy(N_bool_duplicates, N_bool, N_scores)
stop = time.time()
runtimes_file.write(','.join([domain_name, 'HA', str(stop-start)] + [str(x) for x in N_bool.shape]) + '\n')
start = time.time()
gra_syn = gra.find_synergy(N_bool, N_bool, N_scores)
gra_syn_duplicates = gra.find_synergy(N_bool_duplicates, N_bool, N_scores)
stop = time.time()
runtimes_file.write(','.join([domain_name, 'GR', str(stop-start)] + [str(x) for x in N_bool.shape]) + '\n')

pd.DataFrame(N_bool.sum(axis=1), columns=['size']).to_csv(results_data_path + 'coalition_size_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(N_bool_duplicates.sum(axis=1), columns=['size']).to_csv(results_data_path + 'coalition_size_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(simple_synergy, columns=['synergy']).to_csv(results_data_path + 'simple_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(simple_avimp, columns=['avimp']).to_csv(results_data_path + 'simple_' + domain_name + '_avimp.tsv', sep='\t', index=False)
pd.DataFrame(simple_synergy_duplicates, columns=['synergy']).to_csv(results_data_path + 'simple_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(simple_avimp_duplicates, columns=['avimp']).to_csv(results_data_path + 'simple_' + domain_name + '_duplicates_avimp.tsv', sep='\t', index=False)
pd.DataFrame(har_syn, columns=['synergy']).to_csv(results_data_path + 'harsanyi_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(har_syn_duplicates, columns=['synergy']).to_csv(results_data_path + 'harsanyi_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(gra_syn, columns=['synergy']).to_csv(results_data_path + 'grabisch_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(gra_syn_duplicates, columns=['synergy']).to_csv(results_data_path + 'grabisch_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(np.vstack([synergy_values, coalition_values]).transpose(), columns=['synergy', 'colval']).to_csv(
    results_data_path + 'synergy_index_' + domain_name + '.tsv', sep='\t', index=False)
pd.DataFrame(agents_avg_shap, columns=['avshap']).to_csv(results_data_path + 'synergy_index_' + domain_name + '_avshap.tsv', sep='\t', index=False)
pd.DataFrame(np.vstack([synergy_values_duplicates, coalition_values_duplicates]).transpose(), columns=['synergy', 'colval']).to_csv(
    results_data_path + 'synergy_index_' + domain_name + '_duplicates.tsv', sep='\t', index=False)
pd.DataFrame(agents_avg_shap_duplicates, columns=['avshap']).to_csv(results_data_path + 'synergy_index_' + domain_name + '_duplicates_avshap.tsv', sep='\t', index=False)
print('Done!')


#########################
## Close runtimes file ##
#########################
runtimes_file.close()
print('Synergy values for all domains and all indexes found.')
