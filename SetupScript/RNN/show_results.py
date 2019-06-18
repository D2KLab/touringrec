import pandas as pd
import matplotlib.pyplot as plt

df_loss = pd.read_csv('./results/rnn_cuda_sub_loss.csv')
df_acc = pd.read_csv('./results/rnn_cuda_sub_acc.csv')

plt.figure()
plt.plot(df_loss.values.tolist())

plt.figure()
plt.plot(df_acc.values.tolist())
