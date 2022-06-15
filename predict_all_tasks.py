import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,\
    precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE


def random_forest_prediction(train_data, test_data, mode='multi'):
    shuffled_data = train_data.sample(n=len(train_data), random_state=42)
    X_train = ''
    y_train = ''
    X_test = ''
    y_test = ''
    model = ''
    if (mode == 'multi'):
        X_train = shuffled_data.loc[:, 'lfhf':'elapsed-seconds']
        y_train = shuffled_data.loc[:, 'label']
        smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        X_test = test_data.loc[:, 'lfhf':'elapsed-seconds']
        y_test = test_data.loc[:, 'label']
        model = RandomForestClassifier(
                criterion='entropy',
                max_depth=180,
                max_features=28,
                min_samples_split=10,
                n_estimators=36,
                random_state=42
        )
    elif (mode == 'code'):
        X_train = shuffled_data.loc[:, 'sloc':'elapsed-seconds']
        y_train = shuffled_data.loc[:, 'label']
        smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        X_test = test_data.loc[:, 'sloc':'elapsed-seconds']
        y_test = test_data.loc[:, 'label']
        model = RandomForestClassifier(
                criterion='gini',
                max_depth=490,
                max_features=40,
                n_estimators=40,
                min_samples_leaf=1,
                min_samples_split=70,
                random_state=42
        )

    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_test_pred)
    accuracy = round(accuracy_score(y_test, y_test_pred), 3)
    precision = round(precision_score(y_test, y_test_pred), 3)
    recall = round(recall_score(y_test, y_test_pred), 3)
    f1 = round(f1_score(y_test, y_test_pred), 3)
    return [cm, accuracy, precision, recall, f1]


selected_model = 'code'
print('model: ', selected_model)
task1_data = pd.read_csv('./data/merged_task1_features.csv')
task2_data = pd.read_csv('./data/merged_task2_features_fixed.csv')
results1 = random_forest_prediction(task1_data, task2_data, selected_model)
results2 = random_forest_prediction(task2_data, task1_data, selected_model)
total_cm = results1[0] + results2[0]
total_accuracy = round((results1[1] + results2[1]) / 2, 3)
total_precision = round((results1[2] + results2[2]) / 2, 3)
total_recall = round((results1[3] + results2[3]) / 2, 3)
total_f1 = round((results1[4] + results2[4]) / 2, 3)

print('confusion_matrix: ', total_cm)
print('Accuracy: ', total_accuracy)
print('Precision: ', total_precision)
print('Recall: ', total_recall)
print('F1-Score: ', total_f1)
print(total_cm.shape[0])
"""
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(total_cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(total_cm.shape[0]):
    for j in range(total_cm.shape[1]):
        ax.text(x=j, y=i,s=total_cm[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
# plt.title('Confusion Matrix (all params)', fontsize=18)
plt.show()
"""
