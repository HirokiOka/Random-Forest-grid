import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix,\
    precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE


start_col = 'lfhf'
end_col = 'pnn50'

task1_data = pd.read_csv('./data/merged_task1_features.csv')
task2_data = pd.read_csv('./data/merged_task2_features_fixed.csv')

shuffled_data = task1_data.sample(n=len(task1_data))
X_train = shuffled_data.loc[:, start_col:end_col]
y_train = shuffled_data.loc[:, 'label']


smote = SMOTE(sampling_strategy='auto', k_neighbors=5)
X_train, y_train = smote.fit_resample(X_train, y_train)

X_test = task2_data.loc[:, start_col:end_col]
y_test = task2_data.loc[:, 'label']

model = RandomForestClassifier()
cv = KFold(5, shuffle=True)

param_grid = {
        'criterion': ['entropy'],
        'n_estimators': [85],
        'max_depth': [80],
        'max_features': [75],
        'min_samples_leaf': [1],
        'min_samples_split': [10]
        }

grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)
y_pred = grid_search.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ", cm)
print("Accuracy to test: ", round(accuracy_score(y_test, y_pred), 3))
print("Precision Score: ", round(precision_score(y_test, y_pred), 3))
print("Recall Score: ", round(recall_score(y_test, y_pred), 3))
print("F-1 Score: ", round(f1_score(y_test, y_pred), 3))
print(grid_search.best_estimator_, grid_search.best_params_)
