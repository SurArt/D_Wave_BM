#!/usr/bin/env python3

import pandas as pd
from tqdm import tqdm

data = pd.read_csv('data.csv', header=0, sep=';', encoding='cp1251')
data = data[data['event_point'].isin([f"НК-{i + 1}" for i in range(5)])]
data['eventdatetime'] = pd.to_datetime(data['eventdatetime'], format='%Y-%m-%d %H:%M:%S')
data = data.drop(columns=['event_point', 'event_name']).set_index('eventdatetime')
data['rid'] = 1
data = data.resample('5Min').sum().rename(columns={'rid': 'count'})
data['weekday'] = data.index.weekday
data['time'] = data.index
data = data.reset_index()

# discretization
discretization_borders = [max(data['count'])*i/20 for i in range(21)]
# discretization_borders = [
#     0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 19, 24, 29, 34, 39, 44, 49, 54, 59, 69,
#     79, 89, 99, 119, 139, 159, 179, 1000
# ]
DISCR_SIZE = len(discretization_borders) - 1
data['discr_count'] = pd.cut(
    data['count'], discretization_borders,
    labels=[i for i in range(DISCR_SIZE)]
).fillna(0)

# patterns
NUMBER_TICKS = 9
columns = ['prob', 'weekday']
columns.extend([f"{i}" for i in range(NUMBER_TICKS)])
patterns = pd.DataFrame(columns=columns)

for i in tqdm(range(NUMBER_TICKS, len(data))):
    item = data.iloc[i-NUMBER_TICKS:i,:]
    match = patterns['0'] == item['discr_count'].iloc[0]
    for j in range(1, NUMBER_TICKS):
        match &= patterns[str(j)] == item['discr_count'].iloc[j]
    match &= patterns['weekday'] == item['weekday'].iloc[NUMBER_TICKS-1]
    if len(patterns[match]) == 0:
        patterns = patterns.append({
            'prob': 1, 'weekday': item['weekday'].iloc[NUMBER_TICKS-1],
            **{str(j): item['discr_count'].iloc[j] for j in range(NUMBER_TICKS)}
        }, ignore_index=True)
    else:
        patterns.at[patterns[match].index[0], 'prob'] += 1
patterns = patterns['prob'] / sum(patterns['prob'])

# one hot encoding
for i in range(NUMBER_TICKS):
    for j in range(DISCR_SIZE):
        patterns[f"{i}_{j}"] = patterns[f"{i}"].apply(lambda x: 1 if x == j else -1)
patterns = patterns.drop(columns=[f"{i}" for i in range(NUMBER_TICKS)])

for i in range(7):
    patterns[f"weekday_{i}"] = patterns["weekday"].apply(lambda x: 1 if x == i else -1)
patterns = patterns.drop(columns=['weekday'])

patterns.to_csv('bits_with_prob.csv', sep=';', index=False)
