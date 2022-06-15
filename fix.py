import os
import pandas as pd


files = os.listdir('./')

for file in files:
    if ((file == 'fix.py') or (file == '.DS_Store') or (file == 'washino2_feature.csv')):
        continue
    print('file: ', file)
    csv_df = pd.read_csv(file)
    lfhf_list = []
    for idx, val in csv_df['lfhf'].iteritems():
        if (val > 10):
            print(val)
"""
washino_df = pd.read_csv('./washino2_feature.csv')
lfhf = washino_df['lfhf']
new_lfhf = []
for _, val in lfhf.iteritems():
    new_val = val
    if (val >10):
        new_val = 10.0
    new_lfhf.append(new_val)

washino_df['lfhf'] = new_lfhf
washino_df.to_csv('./washino2_feature_fixed.csv')
"""
