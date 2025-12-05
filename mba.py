import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("online_retail_II.csv", encoding="ISO-8859-1")
print("Raw shape:", df.shape)

df = df.rename(columns={
    "Invoice": "Invoice",
    "StockCode": "StockCode",
    "Description": "Description",
    "Quantity": "Quantity",
    "InvoiceDate": "InvoiceDate",
    "Price": "UnitPrice",
    "Customer ID": "CustomerID",
    "Country": "Country"
})


df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["CustomerID"] = df["CustomerID"].astype(str)

df = df.dropna(subset=["InvoiceDate", "CustomerID"])
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]
df = df[~df["Invoice"].astype(str).str.startswith("C")]  # remove cancellations

print("After cleaning:", df.shape)



print("\n=== Building basket for Apriori (using subset + popular items) ===")

df_ap = df[df["Country"] == "United Kingdom"].copy()
print("Rows after UK filter:", df_ap.shape[0])


basket = (
    df_ap.groupby(["Invoice", "Description"])["Quantity"]
         .sum()
         .unstack()
         .fillna(0)
)

print("Raw basket shape (invoices x products):", basket.shape)

item_counts = (basket > 0).sum(axis=0)  
popular_items = item_counts[item_counts >= 50].index
basket = basket[popular_items]

print("Basket shape after filtering popular items:", basket.shape)


basket_sets = basket.gt(0)          
basket_sets = basket_sets.astype(bool)

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


df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
print("\nSnapshot date:", snapshot_date)

rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
    Frequency=("Invoice", "nunique"),
    Monetary=("TotalPrice", "sum")
)

print("\nRFM sample:")
print(rfm.head())

r_q = rfm["Recency"].quantile([0.25, 0.5, 0.75]).to_dict()
f_q = rfm["Frequency"].quantile([0.25, 0.5, 0.75]).to_dict()
m_q = rfm["Monetary"].quantile([0.25, 0.5, 0.75]).to_dict()

def r_score(x):
    if x <= r_q[0.25]: return 4
    elif x <= r_q[0.5]: return 3
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

def assign_segment(row):
    if row["R_Score"] >= 3 and row["F_Score"] >= 3:
        return "Champions"
    elif row["F_Score"] >= 3:
        return "Loyal"
    elif row["R_Score"] == 1:
        return "At Risk"
    else:
        return "Others"

rfm["Segment"] = rfm.apply(assign_segment, axis=1)

print("\nRFM with segments:")
print(rfm.head())


def recommend_for_product(product, rules_df, top_n=5):
    mask = rules_df["antecedents"].apply(lambda x: product in list(x))
    result = rules_df[mask].sort_values("lift", ascending=False)
    if result.empty:
        print(f"No rules found with '{product}' in antecedents.")
        return result
    result = result.copy()
    result["antecedents"] = result["antecedents"].apply(lambda x: list(x))
    result["consequents"] = result["consequents"].apply(lambda x: list(x))
    return result.head(top_n)

print("\nExample cross-sell recommendations:")
try:
    print(recommend_for_product("WHITE CHERRY LIGHTS", rules))
except Exception as e:
    print("Example product not found in rules or other error:", e)


rfm.to_csv("rfm_segments.csv")
rules.to_csv("association_rules.csv")
frequent_itemsets.to_csv("frequent_itemsets.csv")

print("\nSaved: rfm_segments.csv, association_rules.csv, frequent_itemsets.csv")
print("Done.")
