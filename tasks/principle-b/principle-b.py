import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from causal_tasks import causal_models, task_list

TASK = task_list[0]

models = [model.split('/')[1] for model in causal_models]

# reformat data for plotting
gmme = {
    'Model': [],
    'GMME': [],
    'Condition': [],
}

for model in models:
    results = pd.read_csv(f'tasks/{TASK}/results/{TASK}_{model}.csv')

    conds = sorted(results['Condition'].unique())
    data = pd.DataFrame(
        {condition: results[results['Condition'] == condition]['Surprisal'].to_list() for condition in conds}
    )

    gmme['Model'] += [model] * 48
    gmme['GMME'] += list(data['Constraint-Mismatch'] - data['Constraint-Match'])
    gmme['Condition'] += ['Constraint'] * 24
    gmme['GMME'] += list(data['NoConstraint-Mismatch'] - data['NoConstraint-Match'])
    gmme['Condition'] += ['No constraint'] * 24

gmme = pd.DataFrame(gmme)

# draw surprisal bar plots
sns.set_theme(style='darkgrid', rc={'figure.figsize':(6,5), 'figure.dpi': 200})
ax = sns.barplot(
    data=gmme,
    x='Model',
    y='GMME',
    hue='Condition',
    hue_order=('No constraint', 'Constraint'),
    errorbar=None # seaborn's ci95 calculation is off for some reason
)

# add error bars and annotate significance
darkgray = '#424242'

for i, model in enumerate(models):
    conditions = gmme['Condition'].unique()

    data = {condition: gmme[(gmme['Model'] == model) & (gmme['Condition'] == condition)]['GMME'] for condition in conditions}
    means = {const: data[const].mean() for const in conditions}
    ci95 = {
        condition: stats.t.interval(
            .95,
            df=len(data[condition]) - 1,
            loc=means[condition],
            scale=stats.sem(data[condition])
        ) for condition in conditions
    }

    # error bars
    for condition, offset in zip(conditions, [.2, -.2]):
        plt.plot([i + offset] * 2, [*ci95[condition]], color=darkgray, linewidth=2)

    # significance annotation
    y = max(ci95[condition][1] for condition in conditions) + .1

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
        plt.plot([i - .2, i - .2, i + .2, i + .2], [y, y + .1, y + .1, y], color=darkgray, linewidth=1)
        plt.text(i, y + .075, text, horizontalalignment='center')


ax.set(title='Gender mismatch effect by condition', ylabel='Surprisal difference')

# plt.show()
plt.savefig(f'tasks/{TASK}/results/{TASK}.png')
plt.close()
