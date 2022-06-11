import pandas as pd

test_data = pd.read_csv('./data/merged_task2_features.csv')

# print(test_data['lfhf'])

new_lfhf = []
for idx, row in test_data.iterrows():
    lfhf = row['lfhf']
    if (lfhf > 10):
        lfhf = 10
    new_lfhf.append(lfhf)

test_data['lfhf'] = new_lfhf
test_data.to_csv('./data/merged_task2_features_fixed.csv')

