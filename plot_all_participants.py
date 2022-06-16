import os
import pandas as pd
from sklearn.model_selection import train_test_split,\
        cross_val_score, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix,\
    precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from rf_pred import rf_pred

train_data = pd.read_csv('./data/merged_task2_features_fixed.csv')

fig, axes = plt.subplots(5, sharex="all", constrained_layout=True)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["font.size"] = 12
plt.rcParams["scatter.marker"] = ","
plt.rcParams["scatter.edgecolors"] = None
# plt.rcParams['figure.dpi'] = 300

target_dir = './data/paritcipants/anon-task1'
files = os.listdir(target_dir)
files.sort()
i = 0
for filename in files:
    if (filename == '.DS_Store'):
        continue
    file_path = os.path.join(target_dir, filename)
    test_data = pd.read_csv(file_path)
    y_test = test_data.loc[:, 'label']
    y_test_pred = rf_pred(train_data, test_data, mode='code', random_seed=42)

    plot_title = filename.split('.')[0]
    axes[i].set_title(plot_title, fontsize=10)
    axes[i].set_ylim([-0.5, 1.5])
    axes[i].scatter(x=range(len(y_test)), y=y_test+0.15, s=1, c='r', label='Ground-truth')
    axes[i].scatter(x=range(len(y_test_pred)), y=y_test_pred, s=1, c='b', label='Prediction')
    i += 1

lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc='upper right', markerscale=5, ncol=2, bbox_to_anchor=(1, 1.01), borderaxespad=0, fontsize=10)
fig.supxlabel('elapsed seconds[s]')
fig.supylabel('label')
plt.show()
"""
fname = 'code_all_p1.pdf'
plt.savefig(fname, format='pdf')
"""
