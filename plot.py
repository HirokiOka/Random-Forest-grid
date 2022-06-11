import pandas as pd
import matplotlib.pyplot as plt


train_data = pd.read_csv('./data/merged_task1_features.csv')
test_data = pd.read_csv('./data/merged_task2_features_fixed.csv')
X_train = train_data.loc[:, 'lfhf':'pnn50']
X_test = test_data.loc[:, 'lfhf':'pnn50']
plt.plot(X_train, label=list(X_train.columns))
# plt.plot(X_test, label=list(X_train.columns))
plt.legend()
plt.show()
