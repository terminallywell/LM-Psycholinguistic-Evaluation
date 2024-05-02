import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from masked_tasks import masked_models, task_list

TASK = task_list[1]

models = [model.split('/')[1] for model in masked_models]

for model in models:
    data = pd.read_csv(f'tasks/Attraction/results/Attraction_{model}.csv')

    # reformat data for plotting
    il_licit = ['PL', 'SG']
    data_verb = pd.DataFrame(
        {condition: data[(data['Type'] == 'Verbal') & (data['TargetNum'] == condition)]['Surprisal'].to_list()
         for condition in il_licit}
    )
    data_refl = pd.DataFrame(
        {condition: data[(data['Type'] == 'Reflexive') & (data['TargetNum'] == condition)]['Surprisal'].to_list()
         for condition in il_licit}
    )

    nmme = {
        'Type': [],
        'NMME': [],
        'Distractor': [],
    }

    nmme['Type'] += ['Verbal'] * 96
    nmme['NMME'] += list(data_verb['PL'] - data_verb['SG'])
    nmme['Type'] += ['Reflexive'] * 96
    nmme['NMME'] += list(data_refl['PL'] - data_refl['SG'])
    nmme['Distractor'] += ['No intrusion', 'Intrusion'] * 96

    nmme = pd.DataFrame(nmme)

    # draw surprisal difference bar plots
    sns.set_theme(style='darkgrid', rc={'figure.figsize':(6,5), 'figure.dpi': 200})
    ax = sns.barplot(
        data=nmme,
        x='Type',
        y='NMME',
        hue='Distractor',
        errorbar=None
    )
    plt.ylim((0, 12))

    # add error bars and annotate significance
    darkgray = '#424242'

    for i, type in enumerate(['Verbal', 'Reflexive']):
        conditions = nmme['Distractor'].unique()
        
        data = {condition: nmme[(nmme['Type'] == type) & (nmme['Distractor'] == condition)]['NMME'] for condition in conditions}
        means = {condition: data[condition].mean() for condition in conditions}
        ci95 = {
            condition: stats.t.interval(
                .95,
                df=len(data[condition]) - 1,
                loc=means[condition],
                scale=stats.sem(data[condition])
            ) for condition in conditions
        }

        # error bars
        for condition, offset in zip(conditions, [-.2, .2]):
            plt.plot([i + offset] * 2, [*ci95[condition]], color=darkgray, linewidth=2)

        # significance annotation
        y = max(ci95[condition][1] for condition in conditions) + .2

        t, p = stats.ttest_ind(*(data[condition] for condition in conditions))

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
            plt.plot([i - .2, i - .2, i + .2, i + .2], [y, y + .2, y + .2, y], color=darkgray, linewidth=1)
            plt.text(i, y + .15, text, horizontalalignment='center')

    ax.set(title=f'Number mismatch effect by distractor\nModel: {model}', ylabel='Surprisal difference')

    # plt.show()
    plt.savefig(f'tasks/attraction/results/attraction_{model}.png')
    plt.close()
