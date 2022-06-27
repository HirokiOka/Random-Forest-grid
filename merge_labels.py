import pandas as pd

label_A = pd.read_csv('./data/labels/takegawa1_label.csv', header=None)
label_B = pd.read_csv('./data/labels/ohmura1_label.csv', header=None)
"""
start_time = label_A[2]
start_elapsed_sec = label_A[3]
end_time = label_A[4]
end_elapsed_sec = label_A[5]
label_name = label_A[6]

for idx, row in label_A.iterrows():
    start = row[2]
    end = row[4]
    label = row[6]
    print('label:', label, start, '~', end)
"""
