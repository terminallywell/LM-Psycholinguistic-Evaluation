import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from masked_tasks import masked_models


models = [model.split('/')[1] for model in masked_models]

nmme = {
    'Type': [],
    'NMME': [],
    'Distractor': [],
}

for model in [models[2]]: # just roberta base
    results = pd.read_csv(f'results/Attraction_{model}.csv')
    
    il_licit = ['PL', 'SG']
    data_verb = pd.DataFrame(
        {condition: results[(results['Type'] == 'Verbal') & (results['TargetNum'] == condition)]['Surprisal'].to_list()
         for condition in il_licit}
    )

    data_refl = pd.DataFrame(
        {condition: results[(results['Type'] == 'Reflexive') & (results['TargetNum'] == condition)]['Surprisal'].to_list()
         for condition in il_licit}
    )

    # conds = sorted(results['Condition'].unique())
    # data = pd.DataFrame(
    #     {condition: results[results['Condition'] == condition]['Surprisal'].to_list() for condition in conds}
    # )

    # nmme['Model'] += [model] * 192
    nmme['Type'] += ['Verbal'] * 96
    nmme['NMME'] += list(data_verb['PL'] - data_verb['SG'])
    nmme['Type'] += ['Reflexive'] * 96
    nmme['NMME'] += list(data_refl['PL'] - data_refl['SG'])
    nmme['Distractor'] += ['No intrusion', 'Intrusion'] * 96

    # gmme['Model'] += [model] * 48
    # gmme['GMME'] += list(data['Constraint-Mismatch'] - data['Constraint-Match'])
    # gmme['Condition'] += ['Constraint'] * 24
    # gmme['GMME'] += list(data['NoConstraint-Mismatch'] - data['NoConstraint-Match'])
    # gmme['Condition'] += ['No constraint'] * 24

nmme = pd.DataFrame(nmme)


sns.set_theme(style='darkgrid', rc={'figure.figsize':(6,5), 'figure.dpi': 200})
ax = sns.barplot(
    data=nmme,
    x='Type',
    y='NMME',
    hue='Distractor',
    # hue_order=('No constraint', 'Constraint'),
)

# annotate significance
# for i, model in enumerate(models):
#     conditions = nmme['Condition'].unique()

#     data_verb = {condition: nmme[(nmme['Model'] == model) & (nmme['Condition'] == condition)]['GMME'] for condition in conditions}
#     means = {const: data_verb[const].mean() for const in conditions}
#     upper_ci95 = {
#         condition: stats.t.interval(
#             .95,
#             df=len(data_verb[condition]) - 1,
#             loc=means[condition],
#             scale=stats.sem(data_verb[condition])
#         )[1] for condition in conditions
#     }
#     y = max(upper_ci95[condition] for condition in conditions) + .1

#     t, p = stats.ttest_ind(*(data_verb[const] for const in conditions))
#     draw = True
#     if p < .0001:
#         text = '****'
#     elif p < .001:
#         text = '***'
#     elif p < .01:
#         text = '**'
#     elif p < .05:
#         text = '*'
#     else:
#         draw = False
    
#     if draw:
#         plt.plot([i - .2, i - .2, i + .2, i + .2], [y, y + .1, y + .1, y], color='dimgray')
#         plt.text(i, y + .1, text, horizontalalignment='center')


ax.set(title='Number mismatch effect by distractor', ylabel='Surprisal difference')

plt.show()
# plt.savefig('results/PrincipleB.png')
# plt.close()
