import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, pointbiserialr

df = pd.read_csv('data1.csv')

df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

placement_col = None
for c in df.columns:
    if any(x in c for x in ["placed", "status", "placement"]):
        placement_col = c
        break


def normalize_status(x):
    x = str(x).strip().lower()
    if "not" in x or x in ["no", "0", "fail", "nan"]:
        return 0
    elif "placed" in x or x in ["yes", "1", "pass"]:
        return 1
    return np.nan


df["placed"] = df[placement_col].apply(normalize_status)
df = df.dropna(subset=["placed"])
df["placed"] = df["placed"].astype(int)

percent_cols = [c for c in df.columns if any(
    x in c for x in ["percent", "p", "cgpa"]) and c not in [placement_col, "placed"]]

print("Detected percentage-related columns:", percent_cols)


if "degree_p" in df.columns:
    percent_col = "degree_p"
elif len(percent_cols) > 0:
    percent_col = percent_cols[0]
else:
    raise ValueError(
        "No percentage column found. Please check dataset column names.")

df = df.dropna(subset=[percent_col])

placed_vals = df.loc[df["placed"] == 1, percent_col]
not_placed_vals = df.loc[df["placed"] == 0, percent_col]

t_stat, p_value = ttest_ind(placed_vals, not_placed_vals, equal_var=False)

corr, corr_p = pointbiserialr(df[percent_col], df["placed"])

print(f"\n=== Does {percent_col} Matter for Getting Placed? ===")
print(f"Mean {percent_col} (Placed): {placed_vals.mean():.2f}")
print(f"Mean {percent_col} (Not Placed): {not_placed_vals.mean():.2f}")
print(f"T-test statistic: {t_stat:.3f}, p-value: {p_value:.4f}")
print(f"Point-biserial correlation: {corr:.3f}, p-value: {corr_p:.4f}")

if p_value < 0.05:
    print("\n Result: Percentage significantly influences placement (p < 0.05).")
else:
    print("\n Result: Percentage does not significantly influence placement (p ≥ 0.05).")

plt.figure(figsize=(6, 5))
plt.boxplot([placed_vals, not_placed_vals], labels=[
            "Placed", "Not Placed"], patch_artist=True)
plt.title(f"Boxplot of {percent_col} vs Placement")
plt.ylabel(percent_col)
plt.tight_layout()
plt.show()
