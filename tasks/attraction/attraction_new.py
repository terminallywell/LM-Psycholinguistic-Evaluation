import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from masked_tasks import masked_models, task_list

TASK = task_list[1]

models = [model.split('/')[1] for model in masked_models]

# reformat data for plotting
nmme = {
    'Model': [],
    'Type': [],
    'NMME': [],
    'Distractor': [],
}

for model in models:
    data = pd.read_csv(f'tasks/Attraction/results/Attraction_{model}.csv')

    il_licit = ['PL', 'SG']
    data_verb = pd.DataFrame(
        {condition: data[(data['Type'] == 'Verbal') & (data['TargetNum'] == condition)]['Surprisal'].to_list()
         for condition in il_licit}
    )
    data_refl = pd.DataFrame(
        {condition: data[(data['Type'] == 'Reflexive') & (data['TargetNum'] == condition)]['Surprisal'].to_list()
         for condition in il_licit}
    )

    nmme['Model'] += [model] * 192
    nmme['Type'] += ['Verbal'] * 96
    nmme['NMME'] += list(data_verb['PL'] - data_verb['SG'])
    nmme['Type'] += ['Reflexive'] * 96
    nmme['NMME'] += list(data_refl['PL'] - data_refl['SG'])
    nmme['Distractor'] += ['No intrusion', 'Intrusion'] * 96

nmme = pd.DataFrame(nmme)

# draw surprisal difference bar plots
sns.set_theme(style='darkgrid', rc={'figure.figsize':(6,5), 'figure.dpi': 200})

g = sns.FacetGrid(nmme, col='Model', col_wrap=2)

g.map_dataframe(
    sns.barplot,
    data=nmme,
    x='Type',
    y='NMME',
    hue='Distractor',
    order=nmme['Type'].unique(),
    palette = sns.color_palette()
    # errorbar=None
)
g.set_ylabels('Surprisal difference')
g.set_titles('{col_name}')

g.figure.subplots_adjust(top=.9)
g.figure.suptitle('Number mismatch effect by distractor')

g.axes.flat[0].legend(loc='upper left')

# plt.show()
plt.savefig(f'plots/attraction.png')
plt.close()


##########
# penguins = sns.load_dataset('penguins')

# g = sns.FacetGrid(penguins, col='species')
# g.map_dataframe(
#     sns.barplot,
#     data=penguins,
#     x='island',
#     y='bill_length_mm',
#     hue='sex'
# )
# g.add_legend(loc='upper right')

# plt.savefig('penguins.png')
# plt.close()

# sns.get_dataset_names()
# sns.load_dataset('geyser')
