import json

import pandas as pd
import dill
import os


with open('model/models/logreg_v1.pkl', 'rb') as file:
    model = dill.load(file)

samples = os.listdir('samples')
for i in samples:
    with open(f'samples/{i}') as file:
        sample = json.load(file)[0]

        true_target = sample['target']
        x = pd.DataFrame.from_dict([sample])
        x = x.drop(columns='target')
        print(f"session_id: {x['session_id']}")
        print(f"true target: {true_target}")
        print(f"predict target: {model['model'].predict(x)[0]}")
        print('\n')










