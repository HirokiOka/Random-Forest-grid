from rf_pred import rf_pred
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix,\
    precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import pandas as pd

task1 = pd.read_csv('./data/merged_task1_features.csv')
task2 = pd.read_csv('./data/merged_task2_features_fixed.csv')
y_test1 = task1.loc[:, 'label']
y_test2 = task2.loc[:, 'label']
mode = 'bio'
y_pred_task2 = rf_pred(task1, task2, mode=mode, random_seed=42)
y_pred_task1 = rf_pred(task2, task1, mode=mode, random_seed=42, verbosity=1)


print('recall 1:', round(recall_score(y_test1, y_pred_task1, pos_label=1), 3))
print('recall 0:', round(recall_score(y_test1, y_pred_task1, pos_label=0), 3))

print('recall 1:', round(recall_score(y_test1, y_pred_task1, pos_label=1), 3))
print('recall 0:', round(recall_score(y_test1, y_pred_task1, pos_label=0), 3))

"""
cm1 = confusion_matrix(y_test1, y_pred_task1)
cm2 = confusion_matrix(y_test2, y_pred_task2)
total_cm = cm1 + cm2
disp = ConfusionMatrixDisplay(confusion_matrix=total_cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()
"""
