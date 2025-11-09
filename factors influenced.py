import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr

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

num_cols = df.select_dtypes(include=[np.number]).columns.drop(
    "placed", errors="ignore")

results = []
for col in num_cols:
    try:
        correlation, p_value = pointbiserialr(
            df[col].dropna(), df.loc[df[col].dropna().index, "placed"])
        results.append((col, correlation, p_value))
    except Exception:
        continue

num_corr_df = pd.DataFrame(results, columns=[
                           "Feature", "Correlation", "p_value"]).sort_values("Correlation", ascending=False)

print("\nNumeric Features Correlation with Placement:")
print(num_corr_df)

plt.figure(figsize=(8, 5))
plt.barh(num_corr_df["Feature"], num_corr_df["Correlation"], color="skyblue")
plt.xlabel("Point-biserial Correlation")
plt.title("Numeric Features Correlation with Placement")
plt.tight_layout()
plt.show()
