import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix,\
    precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
# import pickle
# import seaborn as sns
# import matplotlib.pyplot as plt

train_data = pd.read_csv('./data/merged_task1_features.csv')
shuffled_data = train_data.sample(n=len(train_data))
X_train = shuffled_data.loc[:, 'lfhf':'elapsed-seconds']
y_train = shuffled_data.loc[:, 'label']

smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

test_data = pd.read_csv('./data/merged_task2_features.csv')
X_test = test_data.loc[:, 'lfhf':'elapsed-seconds']
y_test = test_data.loc[:, 'label']

model = RandomForestClassifier()

cv = KFold(5, shuffle=True)
param_grid = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [1, 10, 50, 100],
        'max_features': [1, 10, 50, 100, 'sqrt', 'log2', None],
        'min_samples_leaf': [1, 10, 50, 100],
        'min_samples_split': [1, 10, 50, 100],
        'n_estimators': [1, 10, 50, 100],
        'bootstrap': [False, True]
        }
grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

y_pred = grid_search.predict(X_test)
y_train_pred = grid_search.predict(X_train)
cm = confusion_matrix(y_test, y_pred)
print(grid_search.best_estimator_, grid_search.best_params_)
# print(cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy'))

print("Confusion Matrix: ", cm)
print("Accuracy to train: ", accuracy_score(y_train, y_train_pred))
print("Accuracy to test: ", accuracy_score(y_test, y_pred))
"""

print("Precision Score: ", precision_score(y_test, y_pred))
print("Recall Score: ", recall_score(y_test, y_pred))
print("F-1 Score: ", f1_score(y_test, y_pred))
"""
