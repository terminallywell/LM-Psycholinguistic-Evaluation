import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

causal_models = [
    'openai-community/gpt2',
    'openai-community/gpt2-medium',
    'openai-community/gpt2-large',
    'facebook/opt-350m',
    'google/gemma-2b',
]

models = [model.split('/')[1] for model in causal_models]

gmme = {
    'Model': [],
    'GMME': [],
    'Context': [],
}

for model in models:
    results = pd.read_csv(f'results/PrincipleB_{model}.csv')

    conds = sorted(results['Condition'].unique())
    data = pd.DataFrame(
        {condition: results[results['Condition'] == condition]['Surprisal'].to_list() for condition in conds}
    )

    gmme['Model'].extend([model] * 48)
    gmme['GMME'].extend(list(data['Constraint-Mismatch'] - data['Constraint-Match']))
    gmme['Context'].extend(['Constraint'] * 24)
    gmme['GMME'].extend(list(data['NoConstraint-Mismatch'] - data['NoConstraint-Match']))
    gmme['Context'].extend(['NoConstraint'] * 24)

gmme = pd.DataFrame(gmme)

sns.set_theme(style='darkgrid', rc={'figure.figsize':(6,9), "figure.dpi": 200})
ax = sns.catplot(
    data=gmme,
    kind='box',
    x='Model',
    y='GMME',
    hue='Context',
)
ax.set(title='Gender mismatch effect by context', ylabel='Surprisal difference')
plt.legend(loc='upper left')
plt.show()
ax.savefig('results/PrincipleB.png')
plt.close()

# means = {cond: data[cond].mean() for cond in conds}
# cis = {cond: stats.t.interval(.95, df=len(data) - 1, loc=means[cond], scale=stats.sem(data[cond])) for cond in conds}
# errs = {cond: cis[cond][1] - cis[cond][0] for cond in conds}

# plt.figure()
# plt.xticks(rotation=15)
# plt.bar(conds, means.values(), yerr=errs.values())
# plt.show()

# # calculating GMME for constraint condition
# models = [model.split('/')[1] for model in causal_models]
# x = np.arange(5)
# gmme = {
#     'Model': [],
#     'Constraint': [],
#     'NoConstraint': [],
# }
# err = {
#     'Constraint': [],
#     'NoConstraint': [],
# }

# for model_name in models:
#     results = pd.read_csv(f'results/PrincipleB_{model_name}.csv')
#     conds = sorted(results['Condition'].unique())
#     data = pd.DataFrame(
#         {condition: results[results['Condition'] == condition]['Surprisal'].to_list() for condition in conds}
#     )
#     gmme['Model'].append(model_name)
#     gmme['Constraint'].append(data['Constraint-Mismatch'].mean() - data['Constraint-Match'].mean())
#     gmme['NoConstraint'].append(data['NoConstraint-Mismatch'].mean() - data['NoConstraint-Match'].mean())

# fig, ax = plt.subplots()
# ax.title('Gender mismatch effect by context')
# ax.grid()
# ax.set_xlabel('Model')
# ax.set_ylabel('Surprisal difference')
# ax.bar(x - 0.2, gmme['Constraint'], width=.4, color='#979797')
# ax.bar(x + 0.2, gmme['NoConstraint'], width=.4, color='#001443')
# # ax.set_xticks(range(5))
# ax.set_xticklabels([''] + models)
# ax.legend(gmme)
# plt.show()

# dat = {models[i]: gmme['Constraint'][i] for i in range(5)}


# plt.figure()
# plt.title('Gender mismatch effect by context')
# ax1 = sns.barplot(gmme, x='Model', y='')
# plt.show()
