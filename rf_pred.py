import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,\
        cross_val_score, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix,\
    precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE


def show_scores(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix: ", cm)
    print("Accuracy to test: ", round(accuracy_score(y_test, y_pred), 3))
    print("Precision Score: ", round(precision_score(y_test, y_pred), 3))
    print("Recall Score: ", round(recall_score(y_test, y_pred), 3))
    print("F-1 Score: ", round(f1_score(y_test, y_pred), 3))


def rf_pred(train_data, test_data, mode='multi', random_seed=None):
    start_col_name = 'lfhf'
    end_col_name = 'elapsed-seconds'
    model = ''
    if (mode == 'code'):
        start_col_name = 'sloc'
    shuffled_data = train_data.sample(n=len(train_data), random_state=random_seed)
    X_train = shuffled_data.loc[:, start_col_name:end_col_name]
    y_train = shuffled_data.loc[:, 'label']

    smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=random_seed)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    X_test = test_data.loc[:, start_col_name:end_col_name]
    y_test = test_data.loc[:, 'label']

    if (mode == 'multi'):
        model = RandomForestClassifier(
                criterion='entropy',
                max_depth=180,
                max_features=28,
                min_samples_split=10,
                n_estimators=36,
                random_state=random_seed
                )
    elif (mode == 'code'):
        model = RandomForestClassifier(
                criterion='gini',
                max_depth=490,
                max_features=40,
                n_estimators=40,
                min_samples_leaf=1,
                min_samples_split=70,
                random_state=random_seed
                )

    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    show_scores(y_test, y_test_pred)
    return y_test_pred


def main():
    task1_data = pd.read_csv('./data/merged_task1_features.csv')
    task2_data = pd.read_csv('./data/merged_task2_features_fixed.csv')
    rf_pred(train_data=task1_data, test_data=task2_data, mode='code', random_seed=42)


if __name__ == '__main__':
    main()
