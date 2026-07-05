import numpy as np
import pandas as pd


def preprocess_data(df):
    """
    Clean Online Retail dataset.
    """

    df = df.dropna(subset=["CustomerID"])

    df = df[
        ~df["InvoiceNo"]
        .astype(str)
        .str.startswith("C")
    ]

    df = df[df["Quantity"] > 0]

    df = df[df["UnitPrice"] > 0]

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    df["CustomerID"] = df["CustomerID"].astype(int)

    df["TotalAmount"] = (
        df["Quantity"] *
        df["UnitPrice"]
    )

    return df


def create_rfm(df):

    snapshot_date = (
        df["InvoiceDate"].max()
        + pd.Timedelta(days=1)
    )

    rfm = (
        df.groupby("CustomerID")
        .agg({
            "InvoiceDate":
            lambda x: (
                snapshot_date - x.max()
            ).days,

            "InvoiceNo":
            "nunique",

            "TotalAmount":
            "sum"
        })
        .reset_index()
    )

    rfm.columns = [
        "CustomerID",
        "Recency",
        "Frequency",
        "Monetary"
    ]

    return rfm


def recommend_product(
        similarity,
        product_name,
        n=5
):

    if product_name not in similarity.index:

        return []

    recommendations = (

        similarity.loc[product_name]

        .sort_values(
            ascending=False
        )

        .iloc[1:n+1]

    )

    return recommendations
  
