import pandas as pd

stimuli = pd.read_csv('misc/giskeskush.csv')

stimuli['Context'] = stimuli.apply(lambda row: row['Context'].partition(row['Target'])[0].strip(), axis=1)

stimuli['Condition'] = stimuli.apply(lambda row: row['Role'] + '-' + row['Gender'], axis=1)

stimuli['Target'] = ['Michael'] * 28 + ['Jonathan'] * 28 + ['Jessica'] * 28 + ['Rachel'] * 28

stimuli = stimuli[['Item', 'Condition', 'Context', 'Target']]

stimuli.to_csv('tasks/Sbj-Obj.csv', index=False)


stimuli = pd.read_csv('tasks/Sbj-Obj.csv')

lens = stimuli['Context'].apply(lambda s: len(s.split())).to_list()

for i in range(0, len(stimuli) - 1, 2):
    for pair in zip(stimuli['Context'][i].split(), stimuli['Context'][i + 1].split()):
        if pair[0] != pair[1]:
            print(pair[0])

for name in stimuli['Target'].unique():
    tokenizer.tokenize(name)

