import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('data1.csv')
df['status'] = df['status'].astype(str).str.strip()

degree_placement = (
    df.groupby("degree_t")["status"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
)

degree_placement_pct = degree_placement * 100
degree_placement_pct = degree_placement_pct.sort_values(
    by="Placed", ascending=False)

print("\nPlacement percentage by degree specialization (sorted by Placed):")
print(degree_placement_pct.round(2))

degree_counts = df['degree_t'].value_counts()
print("\nNumber of samples per degree specialization:")
print(degree_counts)

plt.figure(figsize=(9, 5))
sns.set_style("whitegrid")

if "Placed" not in degree_placement_pct.columns and "Placed" not in degree_placement.columns:
    placed_col = next(
        (c for c in degree_placement_pct.columns if c.lower() == "placed"), None)
else:
    placed_col = "Placed"

if placed_col is None:
    raise ValueError("Could not find a 'Placed' column in 'status' values. Found: " +
                     ", ".join(degree_placement_pct.columns))

ax = sns.barplot(x=degree_placement_pct.index,
                 y=degree_placement_pct[placed_col])
ax.set_title("Placement Rate by Degree Specialization")
ax.set_ylabel("Placement Percentage (%)")
ax.set_xlabel("Degree Specialization")
plt.xticks(rotation=25, ha="right")

for p in ax.patches:
    height = p.get_height()
    ax.annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
