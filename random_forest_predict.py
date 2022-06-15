import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,\
        cross_val_score, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix,\
    precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

train_data = pd.read_csv('./data/merged_task1_features.csv')
shuffled_data = train_data.sample(n=len(train_data), random_state=42)
X_train = shuffled_data.loc[:, 'lfhf':'elapsed-seconds']
y_train = shuffled_data.loc[:, 'label']
print("train positive: ", len(y_train[y_train == 1]), "train negative: ", len(y_train[y_train == 0]))

smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

test_data = pd.read_csv('./data/merged_task2_features_fixed.csv')
X_test = test_data.loc[:, 'lfhf':'elapsed-seconds']
y_test = test_data.loc[:, 'label']
print("test positive: ", len(y_test[y_test == 1]), "test negative: ", len(y_test[y_test == 0]))

model = RandomForestClassifier(
        criterion='entropy',
        max_depth=180,
        max_features=28,
        min_samples_split=10,
        n_estimators=36,
        random_state=42
        )

model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

y_test_list = list(y_test)
y_test_pred_list = list(y_test_pred)
y_and = []
for idx in range(len(y_test_list)):
    test_val = y_test_list[idx]
    pred_val = y_test_pred_list[idx]
    if (test_val == pred_val):
        y_and.append(test_val)
    else:
        y_and.append(-1)

plt.scatter(x=range(len(y_test_list)), y=y_and, s=0.1, c='g', alpha=1)
plt.legend(fontsize=18)
plt.ylim([-0.1, 1.1])
plt.show()

cm = confusion_matrix(y_test, y_test_pred)

print("Confusion Matrix: ", cm)
print("Accuracy to test: ", round(accuracy_score(y_test, y_test_pred), 2))
print("Accuracy to train: ", round(accuracy_score(y_train, y_train_pred), 2))
print("Precision Score: ", round(precision_score(y_test, y_test_pred), 2))
print("Recall Score: ", round(recall_score(y_test, y_test_pred), 2))
print("F-1 Score: ", round(f1_score(y_test, y_test_pred), 2))

"""
plt.xlabel('data index', {"fontsize": 18})
plt.ylabel('label', {"fontsize": 18})
plt.yticks([0, 1])
plt.scatter(x=range(len(y_test_pred)), y=y_test_pred+0.01, s=0.1, c='b', alpha=1, label='prediction')
plt.scatter(x=range(len(y_test)), y=y_test, s=0.1, c='r', alpha=1, label='desired')
plt.legend(fontsize=18)
plt.show()
"""
