import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data1.csv')
percentage_columns = ['ssc_p', 'hsc_p', 'degree_p', 'mba_p']
avg_percentages = df.groupby('status')[percentage_columns].mean()
plt.figure(figsize=(10, 6))
avg_percentages.plot(kind='bar')
plt.legend(title='Percentage Types')
plt.xlabel('Placement Status')
plt.ylabel('Average Percentage')
plt.title('Average Percentages (SSC, HSC, Degree, MBA) vs Placement Status')
plt.tight_layout()
plt.show()
