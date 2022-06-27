import pandas as pd
from rf_pred import rf_pred


train_data = pd.read_csv('./data/merged_task1_features.csv')
test_data = pd.read_csv('./data/merged_task2_features_fixed.csv')
y_test = test_data.loc[:, 'label']
y_pred = rf_pred(train_data, test_data, mode="multi", random_seed=42)
y_code_pred = rf_pred(train_data, test_data, mode="code", random_seed=42)


y_test_list = list(y_test)
y_pred_list = list(y_pred)
y_code_list = list(y_code_pred)

chunck_list = [[], [], [], [], [], [], [], [], [], []]
chunck_id = 0
last_val = 0
for idx in range(len(y_test_list)):
    curernt_val = y_test_list[idx]
    if (curernt_val == 1):
        chunck_list[chunck_id].append(idx)
    if ((last_val == 1) and (curernt_val == 0)):
        chunck_id += 1
    last_val = y_test_list[idx]

multi_result = []
print("multi")
for c_list in chunck_list:
    count = 0
    for idx in c_list:
        if (y_pred_list[idx] == 1):
            count += 1
    multi_result.append(count / len(c_list))
    # print(round(count / len(c_list), 2))


code_result = []
print("code")
for c_list in chunck_list:
    count = 0
    for idx in c_list:
        if (y_code_list[idx] == 1):
            count += 1
    # print(round(count / len(c_list), 2))
    code_result.append(count / len(c_list))

print('code', sum(code_result) / 10)
print('multi', sum(multi_result) / 10)
