import json
import pandas as pd

# List of JSON files
json_files = ['1.json', '2.json', '3.json', '4.json', '5.json', '6.json', '7.json', '8.json', '9.json', '10.json']

def load_and_explode_json(file_name):
    with open("./dataset_basic_advanced_TLN2023/" + file_name, 'r') as f:
        data = json.load(f)

    synsets = data['dataset']
    answers = data['answers']
    is_hard = data['isHard']
    time_diffs = data['timeDiffs']

    rows = []
    for synset, answer, hard, time_diff in zip(synsets, answers, is_hard, time_diffs):
        synset_name = synset.split(':')[0].split("('")[1].split("')")[0]
        rows.append({
            'Synset': synset_name,
            'isBasic_percentage': 1 if answer == 'basic' else 0,
            'IsNotHard_percentage': 1 if not hard else 0,
            'TimeDiffs_sum': time_diff,
        })

    return rows

# Load and merge the datasets
rows = []
for file in json_files:
    rows.extend(load_and_explode_json(file))

# Create the pandas DataFrame
df = pd.DataFrame(rows).groupby('Synset').agg({
    'isBasic_percentage': 'sum',
    'IsNotHard_percentage': 'sum',
    'TimeDiffs_sum': 'sum',
})

df['isBasic_percentage'] = df['isBasic_percentage'] / 10
df['IsNotHard_percentage'] = df['IsNotHard_percentage'] / 10
df['TimeDiffs_average'] = df['TimeDiffs_sum'] /10

# Print the dataframe
print(df)
