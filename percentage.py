import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data1.csv')

avg_degree = df.groupby('status')['degree_p'].mean()

plt.figure()
avg_degree.plot(kind='bar')
plt.xlabel('Placement Status')
plt.ylabel('Average Degree Percentage')
plt.title('Average Degree Percentage vs Placement Status')
plt.tight_layout()


plt.show()

