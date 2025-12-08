# ============================================================
# Online Retail II – Apriori, RFM Segmentation, Cross-Selling
# (Optimized, Memory-Safe & With Cross-Sell Outputs)
# ============================================================

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
# import nagateja as nt


df = pd.read_csv("online_retail_II.csv", encoding="ISO-8859-1")
print("Raw shape:", df.shape)



df = df.rename(columns={
    "Invoice": "Invoice",
    "StockCode": "StockCode",
    "Description": "Description",
    "Quantity": "Quantity",
    "InvoiceDate": "InvoiceDate",
    "Price": "UnitPrice",
    "Customer ID": "CustomerID",   # FIXED
    "Country": "Country"
})


df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["CustomerID"] = df["CustomerID"].astype(str)

df = df.dropna(subset=["InvoiceDate", "CustomerID"])
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]
df = df[~df["Invoice"].astype(str).str.startswith("C")]

print("After cleaning:", df.shape)


# 4. APRIORI


print("\n=== Building basket for Apriori (filtered & optimized) ===")


df_ap = df[df["Country"] == "United Kingdom"].copy()
print("Rows after UK filter:", df_ap.shape[0])

basket = (
    df_ap.groupby(["Invoice", "Description"])["Quantity"]
         .sum()
         .unstack()
         .fillna(0)
)

print("Raw basket shape:", basket.shape)


item_counts = (basket > 0).sum(axis=0)
popular_items = item_counts[item_counts >= 50].index
basket = basket[popular_items]

print("Basket shape after filtering popular items:", basket.shape)


basket_sets = basket.gt(0).astype(bool)
print("Final basket_sets shape:", basket_sets.shape)


frequent_itemsets = apriori(
    basket_sets,
    min_support=0.02,   
    use_colnames=True,
    max_len=2
)

frequent_itemsets = frequent_itemsets.sort_values("support", ascending=False)
print("\nTop frequent itemsets:")
print(frequent_itemsets.head())

# 4.6 RULE GENERATION
rules = association_rules(
    frequent_itemsets,
    metric="lift",
    min_threshold=1.0
)

rules = rules.sort_values("lift", ascending=False)
print("\nTop association rules:")
print(
    rules.head(10)[["antecedents", "consequents", "support", "confidence", "lift"]]
)


# 5. EXTRA: TOP CROSS-SELL RULE OUTPUTS


print("\n=== TOP 20 CROSS-SELL RULES (by Lift) ===")
top_rules = rules.head(20).copy()

top_rules["antecedents"] = top_rules["antecedents"].apply(list)
top_rules["consequents"] = top_rules["consequents"].apply(list)

print(top_rules[["antecedents", "consequents", "support", "confidence", "lift"]])

# ALL CROSS-SELL PAIRS
print("\n=== ALL CROSS-SELL PAIRS ===")

for idx, row in top_rules.iterrows():
    a = ", ".join(row["antecedents"])
    c = ", ".join(row["consequents"])
    print(f"If customer buys: {a}  →  Recommend: {c}")

# MOST COMMON CONSEQUENTS

print("\n=== MOST COMMON CONSEQUENTS (Recommended Products) ===")

cons_list = rules["consequents"].apply(list)
cons_list = [item for sub in cons_list for item in sub]

cons_freq = pd.Series(cons_list).value_counts().head(20)
print(cons_freq)

# MOST COMMON ANTECEDENTS
print("\n=== MOST COMMON ANTECEDENTS (Trigger Products) ===")

ant_list = rules["antecedents"].apply(list)
ant_list = [item for sub in ant_list for item in sub]

ant_freq = pd.Series(ant_list).value_counts().head(20)
print(ant_freq)

# 6. CROSS-SELL FUNCTION

def recommend_for_product(product, rules_df, top_n=5):
    mask = rules_df["antecedents"].apply(lambda x: product in list(x))
    subset = rules_df[mask]

    if subset.empty:
        print(f"\n⚠ No rules found with '{product}' in antecedents.")
        return pd.DataFrame()

    subset = subset.sort_values("lift", ascending=False).copy()
    subset["antecedents"] = subset["antecedents"].apply(list)
    subset["consequents"] = subset["consequents"].apply(list)

    return subset.head(top_n)[["antecedents", "consequents", "support", "confidence", "lift"]]


# EXAMPLE: Choose ANY product from ant_freq index
example_product = ant_freq.index[0]
print(f"\n=== CROSS-SELL RECOMMENDATION FOR: {example_product} ===")
print(recommend_for_product(example_product, rules))

# RFM SEGMENTATION (on full dataset)

df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
    Frequency=("Invoice", "nunique"),
    Monetary=("TotalPrice", "sum")
)

# --------- RFM scoring ---------
r_q = rfm["Recency"].quantile([0.25, 0.50, 0.75]).to_dict()
f_q = rfm["Frequency"].quantile([0.25, 0.50, 0.75]).to_dict()
m_q = rfm["Monetary"].quantile([0.25, 0.50, 0.75]).to_dict()

def r_score(x):
    if x <= r_q[0.25]: return 4
    elif x <= r_q[0.50]: return 3
    elif x <= r_q[0.75]: return 2
    else: return 1

def fm_score(x, q):
    if x <= q[0.25]: return 1
    elif x <= q[0.50]: return 2
    elif x <= q[0.75]: return 3
    else: return 4

rfm["R_Score"] = rfm["Recency"].apply(r_score)
rfm["F_Score"] = rfm["Frequency"].apply(lambda x: fm_score(x, f_q))
rfm["M_Score"] = rfm["Monetary"].apply(lambda x: fm_score(x, m_q))

rfm["RFM_Score"] = (
    rfm["R_Score"].astype(str)
    + rfm["F_Score"].astype(str)
    + rfm["M_Score"].astype(str)
)

print("\n=== RFM SAMPLE ===")
print(rfm.head())

rfm.to_csv("rfm_segments.csv")
rules.to_csv("association_rules.csv")
frequent_itemsets.to_csv("frequent_itemsets.csv")

print("\nSaved output files successfully!")

