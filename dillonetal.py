import pandas as pd
import re

raw = pd.read_csv('misc/experiment1.csv', header=None)

raw['Sentence'] = raw[0].apply(lambda row: ' '.join(row.split()[2:]))


pattern = r'\b(was|were|himself|herself|themselves)\b'

data = pd.DataFrame()
data['Item'] = raw[0].apply(lambda row: row.split()[1])
data['Type'] = (['Verbal'] * 4 + ['Reflexive'] * 4) * 48
data['TargetNum'] = ['SG', 'SG', 'PL', 'PL'] * 96
data['DistNum'] = ['SG', 'PL'] * 192
data['Context'] = raw['Sentence'].apply(lambda row: re.sub(pattern, '{}', row))
data['Target'] = raw['Sentence'].apply(lambda row: re.search(pattern, row).group())

data.to_csv('tasks/Attraction.csv', index=False)
