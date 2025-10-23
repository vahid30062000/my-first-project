import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset("tips")
print(tips.head())
avg_bill = tips.groupby(['day', 'time'])['total_bill'].mean().reset_index()
print(avg_bill)
max_tip = tips.groupby('smoker')['tip'].max().reset_index()
print(max_tip)
plt.figure(figsize=(8, 5))
sns.barplot(data=avg_bill, x='day', y='total_bill', hue='time', palette='cool')
plt.title('Average Total Bill by Day and Time')
plt.ylabel('Average Total Bill ($)')
plt.show()
plt.figure(figsize=(5, 4))
sns.barplot(data=max_tip, x='smoker', y='tip', palette='mako')
plt.title('Maximum Tip by Smoker Status')
plt.ylabel('Maximum Tip ($)')
plt.show()
