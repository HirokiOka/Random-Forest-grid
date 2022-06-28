import os
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix,\
    precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from rf_pred import rf_pred


task1_data = pd.read_csv('./data/merged_task1_features.csv')
task2_data = pd.read_csv('./data/merged_task2_features_fixed.csv')


fig, axes = plt.subplots(5, sharex="all", constrained_layout=True, figsize=(8, 6))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["font.size"] = 12
plt.rcParams["scatter.marker"] = ","
plt.rcParams["scatter.edgecolors"] = None
# plt.subplots_adjust(hspace=0.4)
# plt.rcParams["figure.subplot.hspace"] = 0.9
plt.rcParams["legend.labelspacing"] = 0.1
# plt.rcParams["legend.borderpad"] = 0
plt.rcParams["legend.columnspacing"] = 0.1
# plt.rcParams["legend.handletextpad"] = 0

task1_dir = './data/paritcipants/anon-task1'
task2_dir = './data/paritcipants/anon-task2'

target_dir = task1_dir
files = os.listdir(target_dir)
files.sort()
i = 0
for filename in files:
    if (filename == '.DS_Store'):
        continue
    file_path = os.path.join(target_dir, filename)
    test_data = pd.read_csv(file_path)

    y_test = test_data.loc[:, 'label']
    y_multi_pred = rf_pred(task2_data, test_data, mode='multi', random_seed=42)
    y_code_pred = rf_pred(task2_data, test_data, mode='code', random_seed=42)
    y_bio_pred = rf_pred(task2_data, test_data, mode='bio', random_seed=42)

    plot_title = filename.split('.')[0]
    axes[i].set_title(plot_title, fontsize=10)
    axes[i].set_ylim([-0.2, 1.5])
    axes[i].scatter(x=range(len(y_multi_pred)), y=y_multi_pred+0.3, s=1, c='black', label='Multimodal')
    axes[i].scatter(x=range(len(y_code_pred)), y=y_code_pred+0.2, s=1, c='blue', label='Code-related')
    axes[i].scatter(x=range(len(y_bio_pred)), y=y_bio_pred+0.1, s=1, c='green', label='Biometric')
    axes[i].scatter(x=range(len(y_test)), y=y_test, s=1, c='r', label='Actual')
    i += 1

lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc='upper right', markerscale=3, ncol=4, borderaxespad=0, bbox_to_anchor=(0.995, 1), fontsize=10, handletextpad=0, borderpad=0)
fig.supxlabel('Time[s]')
fig.supylabel('label')
plt.show()
