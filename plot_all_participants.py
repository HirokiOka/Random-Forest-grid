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

target_dir = './data/paritcipants/anon-task2'
files = os.listdir(target_dir)
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
    axes[i].set_title(plot_title, fontsize=12)
    axes[i].set_ylim([-0.5, 1.5])
    axes[i].scatter(x=range(len(y_test)), y=y_test+0.1, s=0.1, c='r', label='Ground-truth')
    axes[i].scatter(x=range(len(y_test_pred)), y=y_test_pred, s=0.1, c='b', label='Prediction')
    i += 1

lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc='center right', markerscale=10)
fig.supxlabel('elapsed seconds[s]')
fig.supylabel('label')
# plt.show()
fname = 'multi_all_p.pdf'
plt.savefig(fname, dpi='figure', format='pdf', metadata='pdf')
