import json
from pprint import pprint

import pandas as pd
import pickle

with open('data/join_data.pkl', 'rb') as file:
    df = pickle.load(file)


for target in range(2):
    for sample_number in range(1, 4):
        df_sml = df[df.target == target]
        sample = df_sml.sample(n=1).to_dict('records')
        with open(f'samples/sample_{sample_number}_{target}.json', 'w') as file:
            json.dump(sample, file, default=str, indent=1)













