__author__ = 'jlow@masdar.ac.ae'

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
import scipy.stats as stats


# Find z-scores for a variable
def normalize(x_):
    return (x_ - np.mean(x_)) / np.std(x_)


def positive_measure(measure, normed=False):
    measure = measure.copy()
    if normed == True:
        measure = (measure - np.mean(measure)) / np.std(measure)
    measure_len = float(len(measure))
    return [np.sum(measure < 0) / measure_len * 100,
            np.sum(measure == 0) / measure_len * 100,
            np.sum(measure > 0) / measure_len * 100]


results_data_path = '/Users/huhwhat/Documents/synergy-indices/ResultsData/'
additive_domain_name_list = ['movies_gross', 'papers', 'bball']
nonadditive_domain_name_list = ['drinks', 'food', 'perfumes', 'movies_score']
additive_domain_name_nice_list = ['Movies-gross', 'Papers', 'Basketball']
nonadditive_domain_name_nice_list = ['Drinks', 'Food', 'Perfumes', 'Movies-score']
domain_name_list = nonadditive_domain_name_list + additive_domain_name_list + ['random']
domain_name_nice_list = nonadditive_domain_name_nice_list + additive_domain_name_nice_list + ['Random']
palette_ = sns.color_palette("Set1", len(domain_name_list))

unique_value_count_list = []
color_ix = 0
for domain_name, domain_name_nice in zip(domain_name_list, domain_name_nice_list):
    # fig = plt.figure()
    # gs_ = gridspec.GridSpec(1, 1)
    simple_df = pd.read_csv(results_data_path + 'simple_' + domain_name + '_duplicates.tsv', sep='\t')
    syn_colval_df = pd.read_csv(results_data_path + 'synergy_index_' + domain_name + '_duplicates.tsv', sep='\t')
    har_df = pd.read_csv(results_data_path + 'harsanyi_' + domain_name + '_duplicates.tsv', sep='\t')
    gra_df = pd.read_csv(results_data_path + 'grabisch_' + domain_name + '_duplicates.tsv', sep='\t')
    size_df = pd.read_csv(results_data_path + 'coalition_size_' + domain_name + '_duplicates.tsv', sep='\t')
    # unique_value_count_list.append([har_df['synergy'].nunique(), gra_df['synergy'].nunique(), syn_colval_df['synergy'].nunique(), simple_df['synergy'].nunique()])
    # print(domain_name, positive_measure(simple_df['avimp'], normed=True))
    # print(domain_name, positive_measure(syn_colval_df['avshap'], normed=True))
    # print(domain_name, positive_measure(har_df['synergy'], normed=True))
    # print(domain_name, positive_measure(gra_df['synergy'], normed=True))
    # sns.jointplot('Size distribution', 'Synergy distribution', pd.DataFrame(np.vstack([har_df['synergy'].values, size_df['size'].values]).transpose(), columns=['Synergy distribution', 'Size distribution']), kind='kde', color=palette_[color_ix], size=3)
    # fig.set_size_inches(4, 4)
    # plt.tight_layout()
    # fig.suptitle(domain_name_nice)
    # plt.savefig('/Users/huhwhat/Documents/synergy-indices/Results/harsanyi_size_syn_dist_' + domain_name + '_duplicates.png', dpi=100)
    # plt.close(fig)
    # sns.jointplot('Size distribution', 'Synergy distribution', pd.DataFrame(np.vstack([gra_df['synergy'].values, size_df['size'].values]).transpose(), columns=['Synergy distribution', 'Size distribution']), kind='kde', color=palette_[color_ix], size=3)
    # fig.set_size_inches(4, 4)
    # plt.tight_layout()
    # fig.suptitle(domain_name_nice)
    # plt.savefig('/Users/huhwhat/Documents/synergy-indices/Results/grabisch_size_syn_dist_' + domain_name + '_duplicates.png', dpi=100)
    # plt.close(fig)
    # sns.jointplot('Size distribution', 'Synergy distribution', pd.DataFrame(np.vstack([syn_colval_df['synergy'].values, size_df['size'].values]).transpose(), columns=['Synergy distribution', 'Size distribution']), kind='kde', color=palette_[color_ix], size=3)
    # fig.set_size_inches(4, 4)
    # plt.tight_layout()
    # fig.suptitle(domain_name_nice)
    # plt.savefig('/Users/huhwhat/Documents/synergy-indices/Results/size_syn_dist_' + domain_name + '_duplicates.png', dpi=100)
    # plt.close(fig)
    color_ix += 1
    print('All plots for ' + domain_name + ' saved.')

print(unique_value_count_list)

## Print the percentages of negative and positive values
# domain_name = 'bball'
# for domain_name, domain_name_nice in zip(domain_name_list, domain_name_nice_list):
#     simple_df = pd.read_csv(results_data_path + 'simple_' + domain_name + '_duplicates_avimp.tsv', sep='\t')
#     syn_colval_df = pd.read_csv(results_data_path + 'synergy_index_' + domain_name + '_duplicates_avshap.tsv', sep='\t')
#     print(domain_name_nice, positive_measure(simple_df['avimp'], normed=True))
#     # sns.distplot(simple_df['synergy'])
#     # plt.show()
#     # print(domain_name_nice, positive_measure(syn_colval_df['avshap'], normed=True))
#     # sns.distplot(syn_colval_df['synergy'])
#     # plt.show()

######################################################################################
## Find the relationship between high avimp movie-scores and high avimp movie-gross ##
######################################################################################
movies_score_agent_dict = pd.read_csv(results_data_path + 'agents_dict_' + 'movies_score' + '.tsv', sep='\t')
movies_gross_agent_dict = pd.read_csv(results_data_path + 'agents_dict_' + 'movies_gross' + '.tsv', sep='\t')
movies_score_avshap = pd.read_csv(results_data_path + 'synergy_index_' + 'movies_score' + '_duplicates_avshap.tsv', sep='\t')
movies_gross_avshap = pd.read_csv(results_data_path + 'synergy_index_' + 'movies_gross' + '_duplicates_avshap.tsv', sep='\t')
movies_score_avimp = pd.read_csv(results_data_path + 'simple_' + 'movies_score' + '_duplicates_avimp.tsv', sep='\t')
movies_gross_avimp = pd.read_csv(results_data_path + 'simple_' + 'movies_gross' + '_duplicates_avimp.tsv', sep='\t')

movies_score_avshap['avshap'] = normalize(movies_score_avshap['avshap']).values
movies_gross_avshap['avshap'] = normalize(movies_gross_avshap['avshap']).values
movies_score_avimp['avimp'] = normalize(movies_score_avimp['avimp']).values
movies_gross_avimp['avimp'] = normalize(movies_gross_avimp['avimp']).values
positive_movies_score_avshap_agents_set = set(movies_score_agent_dict[movies_score_avshap['avshap'] > 0]['key'].values)
positive_movies_gross_avshap_agents_set = set(movies_gross_agent_dict[movies_gross_avshap['avshap'] > 0]['key'].values)
positive_movies_score_avimp_agents_set = set(movies_score_agent_dict[movies_score_avimp['avimp'] > 0]['key'].values)
positive_movies_gross_avimp_agents_set = set(movies_gross_agent_dict[movies_gross_avimp['avimp'] > 0]['key'].values)

print(len(movies_score_agent_dict), len(movies_gross_agent_dict))
print(len(positive_movies_score_avshap_agents_set), len(positive_movies_gross_avshap_agents_set))
movies_double_pos_avshap = positive_movies_score_avshap_agents_set.intersection(positive_movies_gross_avshap_agents_set)
print(len(movies_double_pos_avshap))
print(list(movies_double_pos_avshap))
print(len(positive_movies_score_avimp_agents_set), len(positive_movies_gross_avimp_agents_set))
movies_double_pos_avimp = positive_movies_score_avimp_agents_set.intersection(positive_movies_gross_avimp_agents_set)
print(len(movies_double_pos_avimp))
print(list(movies_double_pos_avimp)[:25])
print(len(movies_double_pos_avimp.intersection(movies_double_pos_avshap)))
print(list(movies_double_pos_avimp.intersection(movies_double_pos_avshap)))

movies_agent_set = set(movies_score_agent_dict['key'].values).intersection(set(movies_gross_agent_dict['key'].values))
movies_score_selected_agent_dict = dict([[key_, value_] for key_, value_ in movies_score_agent_dict.values if key_ in movies_agent_set])
movies_gross_selected_agent_dict = dict([[key_, value_] for key_, value_ in movies_gross_agent_dict.values if key_ in movies_agent_set])
movies_avimp_list = []
for agent_ in movies_agent_set:
    score_ix = movies_score_selected_agent_dict[agent_]
    gross_ix = movies_gross_selected_agent_dict[agent_]
    movies_avimp_list.append([agent_, movies_score_avshap['avshap'].values[score_ix],
                              movies_gross_avshap['avshap'].values[gross_ix],
                              movies_score_avimp['avimp'].values[score_ix],
                              movies_gross_avimp['avimp'].values[gross_ix]])
movies_avimp_df = pd.DataFrame(movies_avimp_list, columns=['agent', 'score_avshap', 'gross_avshap', 'score_avimp', 'gross_avimp'])
movies_avimp_normed_df = movies_avimp_df.copy()
avimp_col_list = ['score_avshap', 'gross_avshap', 'score_avimp', 'gross_avimp']
fig = plt.figure()
# gs_ = gridspec.GridSpec(3, 2)
gs_ = gridspec.GridSpec(2, 1)
ax_list = list()
# for i_ in range(6):
for i_ in range(2):
    ax_list.append(fig.add_subplot(gs_[i_]))
for avimp_col in avimp_col_list:
    movies_avimp_normed_df[avimp_col] = normalize(movies_avimp_normed_df[avimp_col].values)
# import itertools
# for i_, combo_ in enumerate(itertools.combinations(avimp_col_list, 2)):
# for i_, combo_ in enumerate([('score_avshap', 'gross_avshap'), ('score_avimp', 'gross_avimp')]):
#     sns.regplot(combo_[0], combo_[1], movies_avimp_normed_df, ax=ax_list[i_], robust=True, label=', '.join(combo_))
#     sns.regplot(combo_[0], combo_[1], movies_avimp_df, ax=ax_list[i_], robust=True, label=', '.join(combo_))
#     ax_list[i_].legend()
sns.regplot('score_avshap', 'gross_avshap', movies_avimp_normed_df, ax=ax_list[0],
            robust=True, label='Success vs. quality,\nSynergy Index',
            scatter_kws={'alpha': 0.5, 's': 4})
sns.regplot('score_avimp', 'gross_avimp', movies_avimp_normed_df, ax=ax_list[1],
            robust=True, label='Success vs. quality,\nBenchmark',
            scatter_kws={'alpha': 0.5, 's': 4})
ax_list[0].set_xlabel('Synergy Index average impact, movies-score (quality)')
ax_list[0].set_ylabel('Synergy Index average impact, movies-gross (success)')
ax_list[1].set_xlabel('Benchmark average impact, movies-score (quality)')
ax_list[1].set_ylabel('Benchmark average impact, movies-gross (success)')
ax_list[0].set_xlim(-0.081, 0.021)
ax_list[0].set_ylim(-0.0501, -0.0301)
ax_list[1].set_xlim(-4, 3)
ax_list[1].set_ylim(-0.61, 0.81)
ax_list[0].legend(markerscale=3)
ax_list[1].legend(markerscale=3)
fig.set_size_inches(8, 8)
gs_.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
plt.savefig('/Users/huhwhat/Documents/synergy-indices/Results/quality_vs_success_all_measures.png', dpi=100)
plt.close(fig)
slope_, intercept_, r_, p_, stderr_ = stats.linregress(movies_avimp_normed_df['gross_avshap'].values, movies_avimp_normed_df['score_avshap'].values)
print(slope_, intercept_, r_, p_)
slope_, intercept_, r_, p_, stderr_ = stats.linregress(movies_avimp_normed_df['gross_avimp'].values, movies_avimp_normed_df['score_avimp'].values)
print(slope_, intercept_, r_, p_)



## Only plot distributions for Synergy Index synergy on a single graph
fig = plt.figure()
gs_ = gridspec.GridSpec(1, 1)
ax_list = list()
ax_list.append(fig.add_subplot(gs_[0]))
color_ix = 0
for domain_name, domain_name_nice in zip(domain_name_list, domain_name_nice_list):
    syn_colval_df = pd.read_csv(results_data_path + 'synergy_index_' + domain_name + '_duplicates.tsv', sep='\t')
    sns.distplot(normalize(syn_colval_df['synergy']), color=palette_[color_ix], ax=ax_list[0], label=domain_name_nice)
    color_ix += 1
    print('Distplot for ' + domain_name + ' saved.')
ax_list[0].set_xlabel('z-scores, Synergy Index synergy, all domains')
ax_list[0].set_ylabel('Density')
ax_list[0].set_xlim(-3.1, 3.1)
ax_list[0].set_ylim(0, 1.3)
ax_list[0].legend()
fig.set_size_inches(8, 8)
gs_.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
# fig.suptitle('Synergy Index value distributions for all domains')
plt.savefig('/Users/huhwhat/Documents/synergy-indices/Results/synergy_index_all_domains_distributions_duplicates.png', dpi=100)
plt.close(fig)
print('All plots saved.')


## Plot distributions for Synergy Index synergy on separate graphs
fig = plt.figure()
gs_ = gridspec.GridSpec(3, 3)
ax_list = list()
for ix in range(8):
    ax_list.append(fig.add_subplot(gs_[ix]))
color_ix = 0
for domain_name, domain_name_nice in zip(domain_name_list, domain_name_nice_list):
    syn_colval_df = pd.read_csv(results_data_path + 'synergy_index_' + domain_name + '_duplicates.tsv', sep='\t')
    sns.distplot(normalize(syn_colval_df['synergy']), color=palette_[color_ix], ax=ax_list[color_ix], label=domain_name_nice, axlabel=False)
    ax_list[color_ix].legend(loc='upper left')
    # if domain_name == 'perfumes':
    #     ax_list[color_ix].set_ylim(0, 0.4)
    # if domain_name == 'movies_score':
    #     ax_list[color_ix].set_ylim(0, 0.45)
    # ax_list[color_ix].set_xlim(-3.1, 3.1)
    # ax_list[color_ix].set_ylim(0, 1.3)
    color_ix += 1
    print('Distplot for ' + domain_name + ' saved.')
# ax_list[0].set_xlabel('z-scores, Synergy Index synergy, all domains')
# ax_list[0].set_ylabel('Density')
# ax_list[0].legend()
fig.text(0.5, 0.02, 'z-scores, Synergy Index synergy, all domains', ha='center')
fig.text(0.04, 0.5, 'Density', va='center', rotation='vertical')
fig.set_size_inches(8, 8)
gs_.tight_layout(fig)
gs_.tight_layout(fig, rect=[0.04, 0.03, 1, 0.95])
# fig.suptitle('Synergy Index value distributions for all domains')
plt.savefig('/Users/huhwhat/Documents/synergy-indices/Results/synergy_index_all_domains_distributions_duplicates_v4.png', dpi=100)
plt.close(fig)
print('All plots saved.')


## Only plot distributions for Synergy Index average impact on a single graph
fig = plt.figure()
gs_ = gridspec.GridSpec(1, 1)
ax_list = list()
ax_list.append(fig.add_subplot(gs_[0]))
color_ix = 0
for domain_name, domain_name_nice in zip(domain_name_list, domain_name_nice_list):
    avshap_df = pd.read_csv(results_data_path + 'synergy_index_' + domain_name + '_duplicates_avshap.tsv', sep='\t')
    if domain_name == 'movies_score':
        sns.distplot(normalize(avshap_df['avshap']), ax=ax_list[0], label=domain_name_nice, color=palette_[color_ix], bins=300, kde_kws={'bw': 0.15})
        # sns.distplot(normalize(avshap_df['avshap']), ax=ax_list[0], label=domain_name_nice, color=palette_[color_ix], bins=300, kde_kws={'kernel': 'cos'})
    elif domain_name == 'movies_gross':
        sns.distplot(normalize(avshap_df['avshap']), ax=ax_list[0], label=domain_name_nice, color=palette_[color_ix], bins=300, kde_kws={'bw': 0.12})
        # sns.distplot(normalize(avshap_df['avshap']), ax=ax_list[0], label=domain_name_nice, color=palette_[color_ix], bins=300, kde_kws={'kernel': 'cos'})
    else:
        sns.distplot(normalize(avshap_df['avshap']), ax=ax_list[0], label=domain_name_nice, color=palette_[color_ix], kde_kws={'bw': 0.12})
        # sns.distplot(normalize(avshap_df['avshap']), ax=ax_list[0], label=domain_name_nice, color=palette_[color_ix], kde_kws={'kernel': 'cos'})
    color_ix += 1
    print('Distplot for ' + domain_name + ' saved.')
ax_list[0].set_xlabel('z-scores, Synergy Index average impact, all domains')
ax_list[0].set_ylabel('Density')
ax_list[0].set_xlim(-3.1, 3.1)
ax_list[0].set_ylim(0, 1.3)
ax_list[0].legend()
fig.set_size_inches(8, 8)
gs_.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
# fig.suptitle('Synergy Index value distributions for all domains')
plt.savefig('/Users/huhwhat/Documents/synergy-indices/Results/synergy_index_avimp_all_domains_distributions_duplicates.png', dpi=100)
plt.close(fig)
print('All plots saved.')



