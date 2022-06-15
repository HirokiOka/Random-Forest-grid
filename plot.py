import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train_data = pd.read_csv('./data/merged_task1_features.csv')
test_data = pd.read_csv('./data/merged_task2_features_fixed.csv')
X_train = train_data.loc[:, 'lfhf':'elapsed-seconds']
X_test = test_data.loc[:, 'lfhf':'elapsed-seconds']

plt.plot(X_train, label=list(X_train.columns))
plt.plot(X_test, label=list(X_test.columns))
plt.legend()
plt.show()

