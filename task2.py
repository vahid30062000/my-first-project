import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

penguins = sns.load_dataset("penguins")
print(penguins.head())
num_cols = penguins.select_dtypes(include='number').dropna()
corr = num_cols.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap - Penguins Dataset")
plt.show()
corr.abs().unstack().sort_values(ascending=False)
