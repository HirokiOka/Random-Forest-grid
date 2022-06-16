import os
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

# code-related modal model
"""
model = RandomForestClassifier(
        criterion='gini',
        max_depth=490,
        max_features=40,
        n_estimators=40,
        min_samples_leaf=1,
        min_samples_split=70,
        random_state=42
        )

"""
# multi-modal model
model = RandomForestClassifier(
        criterion='entropy',
        max_depth=180,
        max_features=28,
        min_samples_split=10,
        n_estimators=36,
        random_state=42
        )

model.fit(X_train, y_train)

fig, axes = plt.subplots(5, sharex="all", constrained_layout=True)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["font.size"] = 12
plt.rcParams["scatter.marker"] = ","
plt.rcParams["scatter.edgecolors"] = None
# plt.rcParams['figure.dpi'] = 300

target_dir = './data/paritcipants/anon-task2'
files = os.listdir(target_dir)
files.sort()
i = 0
for filename in files:
    if (filename == '.DS_Store'):
        continue
    file_path = os.path.join(target_dir, filename)
    test_data = pd.read_csv(file_path)
    X_test = test_data.loc[:, 'lfhf':'elapsed-seconds']
    y_test = test_data.loc[:, 'label']
    y_test_pred = model.predict(X_test)

    plot_title = filename.split('.')[0]
    axes[i].set_title(plot_title, fontsize=10)
    axes[i].set_ylim([-0.5, 1.5])
    axes[i].scatter(x=range(len(y_test)), y=y_test+0.15, s=1, c='r', label='Ground-truth')
    axes[i].scatter(x=range(len(y_test_pred)), y=y_test_pred, s=1, c='b', label='Prediction')
    i += 1

lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc='upper right', markerscale=5, ncol=2, bbox_to_anchor=(1, 1), borderaxespad=0, fontsize=10)
fig.supxlabel('elapsed seconds[s]')
fig.supylabel('label')
# plt.show()
fname = 'multi_all_p.pdf'
plt.savefig(fname, format='pdf')
