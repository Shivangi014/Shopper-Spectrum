# ==========================================================
# Shopper Spectrum - train.py
# ==========================================================

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------
# Create models directory
# ----------------------------------------------------------

os.makedirs("models", exist_ok=True)

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

print("Loading dataset...")

df = pd.read_excel(
    "online_retail (Recovered).xlsb",
    engine="pyxlsb"
)

print(f"Dataset Shape : {df.shape}")

# ----------------------------------------------------------
# Data Cleaning
# ----------------------------------------------------------

print("Cleaning dataset...")

# Remove missing CustomerID
df = df.dropna(subset=["CustomerID"])

# Remove cancelled invoices
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

# Remove invalid Quantity
df = df[df["Quantity"] > 0]

# Remove invalid Price
df = df[df["UnitPrice"] > 0]

# Remove duplicates
df = df.drop_duplicates()

# Convert datatype
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

df["CustomerID"] = df["CustomerID"].astype(int)

# ----------------------------------------------------------
# Feature Engineering
# ----------------------------------------------------------

df["TotalAmount"] = (
    df["Quantity"] *
    df["UnitPrice"]
)

print("Cleaned Dataset Shape :", df.shape)

print(df.head())

# ----------------------------------------------------------
# Dataset Statistics
# ----------------------------------------------------------

print("\nCustomers :", df["CustomerID"].nunique())
print("Products  :", df["Description"].nunique())
print("Countries :", df["Country"].nunique())
print("Revenue   :", round(df["TotalAmount"].sum(), 2))

# ----------------------------------------------------------
# Save cleaned dataset (optional)
# ----------------------------------------------------------

df.to_csv(
    "cleaned_online_retail.csv",
    index=False
)

print("Cleaned dataset saved.")
# ==========================================================
# RFM FEATURE ENGINEERING
# ==========================================================

print("\nCreating RFM Dataset...")

# Snapshot Date
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = (
    df.groupby("CustomerID")
      .agg({
          "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
          "InvoiceNo": "nunique",
          "TotalAmount": "sum"
      })
      .reset_index()
)

rfm.columns = [
    "CustomerID",
    "Recency",
    "Frequency",
    "Monetary"
]

print(rfm.head())

# Save RFM dataset
rfm.to_csv("models/rfm_dataset.csv", index=False)

# ==========================================================
# FEATURE TRANSFORMATION
# ==========================================================

print("\nApplying Log Transformation...")

rfm_model = rfm.copy()

rfm_model["Frequency"] = np.log1p(rfm_model["Frequency"])
rfm_model["Monetary"] = np.log1p(rfm_model["Monetary"])

X = rfm_model[
    ["Recency", "Frequency", "Monetary"]
]

# ==========================================================
# STANDARDIZATION
# ==========================================================

print("Scaling Features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("Scaler Saved.")

# ==========================================================
# ELBOW METHOD
# ==========================================================

print("\nFinding Optimal Clusters...")

inertia = []

K = range(2,11)

for k in K:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertia.append(model.inertia_)

print("\nElbow Method Results")

for k, score in zip(K, inertia):

    print(f"K = {k}   Inertia = {score:.2f}")

# ==========================================================
# SILHOUETTE SCORE
# ==========================================================

print("\nSilhouette Scores")

best_k = None
best_score = -1

for k in range(2,11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    print(
        f"K = {k} --> {score:.4f}"
    )

    if score > best_score:

        best_score = score

        best_k = k

print("\nBest K :", best_k)

# ==========================================================
# FINAL MODEL
# ==========================================================

print("\nTraining Final KMeans...")

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(X_scaled)

# Save model
joblib.dump(
    kmeans,
    "models/kmeans_model.pkl"
)

print("KMeans Model Saved.")

# ==========================================================
# CLUSTER SUMMARY
# ==========================================================

summary = (
    rfm.groupby("Cluster")
       [["Recency","Frequency","Monetary"]]
       .mean()
)

print("\nCluster Summary")

print(summary)

summary.to_csv(
    "models/cluster_summary.csv"
)

# ==========================================================
# SAVE RFM WITH CLUSTERS
# ==========================================================

rfm.to_csv(
    "models/rfm_clusters.csv",
    index=False
)

print("\nRFM Dataset Saved.")
# ==========================================================
# PRODUCT RECOMMENDATION MODEL
# ==========================================================

print("\nBuilding Product Recommendation Model...")

# Keep only required columns
recommendation_df = df[
    [
        "CustomerID",
        "Description",
        "Quantity"
    ]
].copy()

# Remove missing descriptions
recommendation_df = recommendation_df.dropna(
    subset=["Description"]
)

# Remove duplicate customer-product pairs
recommendation_df = (
    recommendation_df
    .groupby(
        ["CustomerID", "Description"]
    )["Quantity"]
    .sum()
    .reset_index()
)

# ==========================================================
# CUSTOMER × PRODUCT MATRIX
# ==========================================================

print("Creating Customer-Product Matrix...")

customer_product_matrix = recommendation_df.pivot_table(
    index="CustomerID",
    columns="Description",
    values="Quantity",
    fill_value=0
)

print(customer_product_matrix.shape)

# ==========================================================
# ITEM-ITEM MATRIX
# ==========================================================

print("Computing Cosine Similarity...")

item_matrix = customer_product_matrix.T

similarity = cosine_similarity(item_matrix)

similarity_df = pd.DataFrame(
    similarity,
    index=item_matrix.index,
    columns=item_matrix.index
)

# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(
    similarity_df,
    "models/similarity.pkl"
)

joblib.dump(
    similarity_df.index.tolist(),
    "models/product_list.pkl"
)

print("Similarity Matrix Saved.")

# ==========================================================
# SAMPLE RECOMMENDATION
# ==========================================================

def recommend(product_name, n=5):

    if product_name not in similarity_df.index:
        return []

    recommendations = (
        similarity_df.loc[product_name]
        .sort_values(ascending=False)
        .iloc[1:n+1]
    )

    return recommendations


sample_product = similarity_df.index[0]

print("\nSample Product:")
print(sample_product)

print("\nTop 5 Recommendations:")

print(recommend(sample_product))

# ==========================================================
# SAVE SAMPLE OUTPUT
# ==========================================================

sample_output = recommend(sample_product)

sample_output.to_csv(
    "models/sample_recommendation.csv"
)

print("\nRecommendation File Saved.")

# ==========================================================
# FINISHED
# ==========================================================

print("\n===================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("===================================")

print("\nGenerated Files:\n")

files = [
    "models/scaler.pkl",
    "models/kmeans_model.pkl",
    "models/similarity.pkl",
    "models/product_list.pkl",
    "models/rfm_dataset.csv",
    "models/rfm_clusters.csv",
    "models/cluster_summary.csv",
    "models/sample_recommendation.csv"
]

for file in files:
    print(file)

print("\nRun the Streamlit app using:")
print("streamlit run app.py")
