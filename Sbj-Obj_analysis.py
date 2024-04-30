import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

causal_models = [
    'openai-community/gpt2',
    'facebook/opt-350m',
    'openai-community/gpt2-medium',
    'openai-community/gpt2-large',
    'google/gemma-2b',
]

models = [model.split('/')[1] for model in causal_models]

gmme = {
    'Model': [],
    'GMME': [],
    'Condition': [],
}

for model in models:
    results = pd.read_csv(f'results/Sbj-Obj_{model}.csv')

    conds = sorted(results['Condition'].unique())
    data = pd.DataFrame(
        {condition: results[results['Condition'] == condition]['Surprisal'].to_list() for condition in conds}
    )

    gmme['Model'] += [model] * 56
    gmme['GMME'] += list(data['SBJ-Mismatch'] - data['SBJ-Match'])
    gmme['Condition'] += ['Subject'] * 28
    gmme['GMME'] += list(data['OBJ-Mismatch'] - data['OBJ-Match'])
    gmme['Condition'] += ['Object'] * 28

gmme = pd.DataFrame(gmme)


sns.set_theme(style='darkgrid', rc={'figure.figsize':(6,5), 'figure.dpi': 200})
ax = sns.barplot(
    data=gmme,
    x='Model',
    y='GMME',
    hue='Condition',
)

# annotate significance
for i, model in enumerate(models):
    conditions = gmme['Condition'].unique()

    data = {condition: gmme[(gmme['Model'] == model) & (gmme['Condition'] == condition)]['GMME'] for condition in conditions}
    means = {const: data[const].mean() for const in conditions}

    upper_ci95 = {
        condition: stats.t.interval(
            .95,
            df=len(data[condition]) - 1,
            loc=means[condition],
            scale=stats.sem(data[condition])
        )[1] for condition in conditions
    }

    y = max(upper_ci95[condition] for condition in conditions) + .1

    t, p = stats.ttest_ind(*(data[const] for const in conditions))
    
    draw = True
    if p < .0001:
        text = '****'
    elif p < .001:
        text = '***'
    elif p < .01:
        text = '**'
    elif p < .05:
        text = '*'
    else:
        draw = False
    
    if draw:
        plt.plot([i - .2, i - .2, i + .2, i + .2], [y, y + .1, y + .1, y], color='dimgray')
        plt.text(i, y + .1, text, horizontalalignment='center')


ax.set(title='Gender mismatch effect by condition', ylabel='Surprisal difference')

# plt.show()
plt.savefig('results/Sbj-Obj.png')
plt.close()
