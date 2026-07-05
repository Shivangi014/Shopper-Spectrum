import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>
.main{
    background-color:#f7f9fc;
}
h1,h2,h3{
    color:#1f4e79;
}
.metric-card{
    background:white;
    padding:15px;
    border-radius:10px;
    box-shadow:0px 0px 8px rgba(0,0,0,.15);
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Load Dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("online_retail (Recovered).xlsb encoding="ISO-8859-1"")

    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

    return df


df = load_data()

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.image(
    "https://img.icons8.com/color/96/shopping-cart.png",
    width=80
)

st.sidebar.title("🛒 Shopper Spectrum")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 EDA Dashboard",
        "👥 Customer Segmentation",
        "🎯 Predict Segment",
        "🛍 Product Recommendation",
        "ℹ About"
    ]
)
# ==========================================================
# HOME PAGE
# ==========================================================

if menu == "🏠 Home":

    st.title("🛒 Shopper Spectrum")
    st.markdown(
        """
        ### Customer Segmentation & Product Recommendation System

        This dashboard analyzes customer purchasing behavior using **RFM Analysis**
        and recommends products using **Item-Based Collaborative Filtering**.
        """
    )

    # -------------------------------
    # KPI Cards
    # -------------------------------

    total_transactions = len(df)
    total_customers = df["CustomerID"].nunique()
    total_products = df["Description"].nunique()
    total_countries = df["Country"].nunique()
    total_revenue = df["TotalAmount"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Transactions",
        f"{total_transactions:,}"
    )

    c2.metric(
        "Customers",
        f"{total_customers:,}"
    )

    c3.metric(
        "Products",
        f"{total_products:,}"
    )

    c4.metric(
        "Countries",
        f"{total_countries:,}"
    )

    c5.metric(
        "Revenue",
        f"${total_revenue:,.2f}"
    )

    st.divider()

    # -------------------------------
    # Dataset Preview
    # -------------------------------

    st.subheader("📄 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.divider()

    # -------------------------------
    # Dataset Information
    # -------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Dataset Information")

        info = pd.DataFrame({
            "Feature": df.columns,
            "Data Type": df.dtypes.astype(str)
        })

        st.dataframe(
            info,
            use_container_width=True
        )

    with col2:

        st.subheader("Missing Values")

        missing = pd.DataFrame({
            "Column": df.columns,
            "Missing": df.isnull().sum().values
        })

        st.dataframe(
            missing,
            use_container_width=True
        )

    st.divider()

    # -------------------------------
    # Revenue by Country
    # -------------------------------

    st.subheader("🌍 Revenue by Country")

    country = (
        df.groupby("Country")["TotalAmount"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        country,
        x="Country",
        y="TotalAmount",
        color="TotalAmount",
        text_auto=".2s",
        title="Top 10 Countries by Revenue"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # -------------------------------
    # Monthly Revenue Trend
    # -------------------------------

    st.subheader("📈 Monthly Revenue Trend")

    monthly = (
        df
        .set_index("InvoiceDate")
        .resample("M")["TotalAmount"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        monthly,
        x="InvoiceDate",
        y="TotalAmount",
        markers=True,
        title="Monthly Revenue"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # -------------------------------
    # Top Products
    # -------------------------------

    st.subheader("🔥 Top Selling Products")

    top_products = (
        df.groupby("Description")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )

    fig = px.bar(
        top_products,
        x="Quantity",
        y="Description",
        orientation="h",
        color="Quantity",
        text_auto=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # -------------------------------
    # Footer
    # -------------------------------

    st.success(
        "Use the navigation menu on the left to explore EDA, "
        "Customer Segmentation and Product Recommendation."
    )
# ==========================================================
# EDA DASHBOARD
# ==========================================================

elif menu == "📊 EDA Dashboard":

    st.title("📊 Exploratory Data Analysis")

    st.markdown("Analyze customer transactions and sales performance interactively.")

    # ---------------------------------
    # Sidebar Filters
    # ---------------------------------

    st.sidebar.subheader("EDA Filters")

    countries = sorted(df["Country"].unique())

    selected_country = st.sidebar.multiselect(
        "Select Country",
        countries,
        default=countries
    )

    filtered_df = df[df["Country"].isin(selected_country)]

    # ---------------------------------
    # Dataset Shape
    # ---------------------------------

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", f"{filtered_df.shape[0]:,}")
    c2.metric("Columns", filtered_df.shape[1])
    c3.metric("Revenue", f"${filtered_df['TotalAmount'].sum():,.2f}")

    st.divider()

    # ---------------------------------
    # Dataset Preview
    # ---------------------------------

    with st.expander("View Dataset"):
        st.dataframe(filtered_df)

    st.divider()

    # ---------------------------------
    # Statistical Summary
    # ---------------------------------

    st.subheader("📋 Statistical Summary")

    st.dataframe(filtered_df.describe())

    st.divider()

    # ---------------------------------
    # Missing Values
    # ---------------------------------

    st.subheader("Missing Values")

    missing = filtered_df.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Values"]

    fig = px.bar(
        missing,
        x="Column",
        y="Missing Values",
        color="Missing Values",
        text_auto=True
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------------------------------
    # Top Countries
    # ---------------------------------

    st.subheader("🌍 Top Countries")

    country_sales = (
        filtered_df.groupby("Country")["TotalAmount"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        country_sales,
        x="Country",
        y="TotalAmount",
        color="TotalAmount",
        text_auto=".2s"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------------------------------
    # Top Products
    # ---------------------------------

    st.subheader("🔥 Top Selling Products")

    products = (
        filtered_df.groupby("Description")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
        .reset_index()
    )

    fig = px.bar(
        products,
        x="Quantity",
        y="Description",
        orientation="h",
        color="Quantity"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------------------------------
    # Monthly Revenue
    # ---------------------------------

    st.subheader("📈 Monthly Revenue")

    monthly = (
        filtered_df
        .set_index("InvoiceDate")
        .resample("M")["TotalAmount"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        monthly,
        x="InvoiceDate",
        y="TotalAmount",
        markers=True
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------------------------------
    # Daily Revenue
    # ---------------------------------

    st.subheader("📅 Daily Revenue")

    daily = (
        filtered_df
        .set_index("InvoiceDate")
        .resample("D")["TotalAmount"]
        .sum()
        .reset_index()
    )

    fig = px.area(
        daily,
        x="InvoiceDate",
        y="TotalAmount"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------------------------------
    # Revenue Distribution
    # ---------------------------------

    st.subheader("Revenue Distribution")

    fig = px.histogram(
        filtered_df,
        x="TotalAmount",
        nbins=60
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------------------------------
    # Quantity Distribution
    # ---------------------------------

    st.subheader("Quantity Distribution")

    fig = px.histogram(
        filtered_df,
        x="Quantity",
        nbins=50
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------------------------------
    # Unit Price Distribution
    # ---------------------------------

    st.subheader("Unit Price Distribution")

    fig = px.box(
        filtered_df,
        y="UnitPrice"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------------------------------
    # Correlation Heatmap
    # ---------------------------------

    st.subheader("Correlation Heatmap")

    corr = filtered_df[
        ["Quantity", "UnitPrice", "TotalAmount"]
    ].corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Viridis"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------------------------------
    # Top Customers
    # ---------------------------------

    st.subheader("💰 Top Customers")

    customers = (
        filtered_df.groupby("CustomerID")["TotalAmount"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )

    fig = px.bar(
        customers,
        x="CustomerID",
        y="TotalAmount",
        color="TotalAmount",
        text_auto=".2s"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------------------------------
    # Purchase Hour Analysis
    # ---------------------------------

    filtered_df["Hour"] = filtered_df["InvoiceDate"].dt.hour

    hour = (
        filtered_df.groupby("Hour")
        .size()
        .reset_index(name="Transactions")
    )

    st.subheader("🕒 Purchase Hour Analysis")

    fig = px.line(
        hour,
        x="Hour",
        y="Transactions",
        markers=True
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------------------------------
    # Weekday Analysis
    # ---------------------------------

    filtered_df["Weekday"] = (
        filtered_df["InvoiceDate"]
        .dt.day_name()
    )

    weekday = (
        filtered_df.groupby("Weekday")
        .size()
        .reset_index(name="Transactions")
    )

    order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    weekday["Weekday"] = pd.Categorical(
        weekday["Weekday"],
        categories=order,
        ordered=True
    )

    weekday = weekday.sort_values("Weekday")

    st.subheader("📆 Transactions by Weekday")

    fig = px.bar(
        weekday,
        x="Weekday",
        y="Transactions",
        color="Transactions"
    )

    st.plotly_chart(fig, use_container_width=True)
# ==========================================================
# CUSTOMER SEGMENTATION
# ==========================================================

elif menu == "👥 Customer Segmentation":

    st.title("👥 Customer Segmentation")

    st.markdown("""
    Customer segmentation is performed using **RFM Analysis**
    and **KMeans Clustering**.
    """)

    # -----------------------------
    # RFM Calculation
    # -----------------------------

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

    st.subheader("RFM Dataset")

    st.dataframe(rfm.head())

    st.divider()

    # ---------------------------------
    # RFM Summary
    # ---------------------------------

    st.subheader("RFM Statistics")

    st.dataframe(rfm.describe())

    st.divider()

    # ---------------------------------
    # Histograms
    # ---------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:

        fig = px.histogram(
            rfm,
            x="Recency",
            nbins=40,
            title="Recency Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    with c2:

        fig = px.histogram(
            rfm,
            x="Frequency",
            nbins=40,
            title="Frequency Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    with c3:

        fig = px.histogram(
            rfm,
            x="Monetary",
            nbins=40,
            title="Monetary Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---------------------------------
    # Load Trained Model
    # ---------------------------------

    try:

        scaler = joblib.load("models/scaler.pkl")
        kmeans = joblib.load("models/kmeans_model.pkl")

        X = rfm[["Recency", "Frequency", "Monetary"]].copy()

        X["Frequency"] = np.log1p(X["Frequency"])
        X["Monetary"] = np.log1p(X["Monetary"])

        X_scaled = scaler.transform(X)

        rfm["Cluster"] = kmeans.predict(X_scaled)

    except:

        st.warning(
            "Train the model first using train.py "
            "to visualize customer clusters."
        )

        st.stop()

    st.subheader("Cluster Distribution")

    cluster_count = (
        rfm["Cluster"]
        .value_counts()
        .reset_index()
    )

    cluster_count.columns = [
        "Cluster",
        "Customers"
    ]

    fig = px.pie(
        cluster_count,
        names="Cluster",
        values="Customers",
        hole=0.5
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ---------------------------------
    # Cluster Scatter
    # ---------------------------------

    st.subheader("Customer Clusters")

    fig = px.scatter(

        rfm,

        x="Recency",

        y="Monetary",

        color=rfm["Cluster"].astype(str),

        size="Frequency",

        hover_data=["CustomerID"],

        title="RFM Customer Segments"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.divider()

    # ---------------------------------
    # Cluster Profiles
    # ---------------------------------

    st.subheader("Cluster Profiles")

    profile = (

        rfm.groupby("Cluster")[

            [

                "Recency",

                "Frequency",

                "Monetary"

            ]

        ]

        .mean()

        .round(2)

    )

    st.dataframe(

        profile,

        use_container_width=True

    )

    st.divider()

    # ---------------------------------
    # Download RFM
    # ---------------------------------

    csv = rfm.to_csv(index=False)

    st.download_button(

        "⬇ Download RFM Dataset",

        csv,

        file_name="rfm_dataset.csv",

        mime="text/csv"

    )
# ==========================================================
# PREDICT CUSTOMER SEGMENT
# ==========================================================

elif menu == "🎯 Predict Segment":

    st.title("🎯 Predict Customer Segment")

    st.markdown("""
    Enter the customer's **Recency, Frequency, and Monetary (RFM)** values
    to predict the customer segment.
    """)

    st.divider()

    # -------------------------
    # User Inputs
    # -------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        recency = st.number_input(
            "Recency (Days)",
            min_value=0,
            value=30,
            step=1
        )

    with col2:
        frequency = st.number_input(
            "Frequency",
            min_value=1,
            value=5,
            step=1
        )

    with col3:
        monetary = st.number_input(
            "Monetary",
            min_value=0.0,
            value=500.0,
            step=10.0
        )

    st.divider()

    if st.button("Predict Customer Segment", use_container_width=True):

        try:

            scaler = joblib.load("models/scaler.pkl")
            model = joblib.load("models/kmeans_model.pkl")

            X = pd.DataFrame({
                "Recency":[recency],
                "Frequency":[np.log1p(frequency)],
                "Monetary":[np.log1p(monetary)]
            })

            X_scaled = scaler.transform(X)

            cluster = int(model.predict(X_scaled)[0])

            labels = {
                0: "High-Value",
                1: "Regular",
                2: "Occasional",
                3: "At-Risk"
            }

            colors = {
                "High-Value":"🟢",
                "Regular":"🔵",
                "Occasional":"🟡",
                "At-Risk":"🔴"
            }

            segment = labels.get(cluster, f"Cluster {cluster}")

            st.success(
                f"{colors.get(segment,'⚪')} Customer Segment: **{segment}**"
            )

            st.divider()

            # -------------------------
            # Segment Description
            # -------------------------

            descriptions = {

                "High-Value":
                """
                ### 🟢 High-Value Customer

                - Frequent buyer
                - Recently purchased
                - High spending customer
                - Eligible for loyalty rewards
                """,

                "Regular":
                """
                ### 🔵 Regular Customer

                - Purchases consistently
                - Moderate spending
                - Can be converted into High-Value
                """,

                "Occasional":
                """
                ### 🟡 Occasional Customer

                - Shops infrequently
                - Low spending
                - Responds well to promotions
                """,

                "At-Risk":
                """
                ### 🔴 At-Risk Customer

                - Hasn't purchased recently
                - Low engagement
                - Needs retention campaigns
                """
            }

            st.info(descriptions.get(segment))

        except Exception as e:

            st.error("Model files not found.")

            st.code(str(e))
# ==========================================================
# PRODUCT RECOMMENDATION
# ==========================================================

elif menu == "🛍 Product Recommendation":

    st.title("🛍 Product Recommendation System")

    st.markdown("""
    Recommend similar products using **Item-Based Collaborative Filtering**
    based on customers' purchasing behavior.
    """)

    st.divider()

    try:

        similarity = joblib.load("models/similarity.pkl")

    except Exception as e:

        st.error("❌ similarity.pkl not found. Run train.py first.")

        st.code(str(e))

        st.stop()

    # ----------------------------------------
    # Product Selection
    # ----------------------------------------

    product_list = sorted(similarity.index.tolist())

    product = st.selectbox(
        "Select a Product",
        product_list
    )

    top_n = st.slider(
        "Number of Recommendations",
        min_value=1,
        max_value=10,
        value=5
    )

    # ----------------------------------------
    # Recommendation Function
    # ----------------------------------------

    def recommend_products(product_name, n=5):

        scores = similarity.loc[product_name]

        scores = scores.sort_values(
            ascending=False
        )

        scores = scores.iloc[1:n+1]

        return scores

    # ----------------------------------------
    # Recommend Button
    # ----------------------------------------

    if st.button(
        "Get Recommendations",
        use_container_width=True
    ):

        recommendations = recommend_products(
            product,
            top_n
        )

        st.success(
            f"Top {top_n} products similar to **{product}**"
        )

        st.divider()

        for i, (item, score) in enumerate(
            recommendations.items(),
            start=1
        ):

            with st.container(border=True):

                c1, c2 = st.columns([4,1])

                with c1:

                    st.markdown(
                        f"### {i}. {item}"
                    )

                with c2:

                    st.metric(
                        "Similarity",
                        f"{score:.2f}"
                    )

        st.divider()

        # Similarity Chart

        fig = px.bar(
            x=recommendations.values,
            y=recommendations.index,
            orientation="h",
            text=np.round(
                recommendations.values,
                2
            ),
            labels={
                "x":"Similarity Score",
                "y":"Products"
            },
            title="Recommended Products"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # Download CSV

        csv = recommendations.reset_index()

        csv.columns = [
            "Product",
            "Similarity"
        ]

        st.download_button(
            "⬇ Download Recommendations",
            csv.to_csv(index=False),
            file_name="recommendations.csv",
            mime="text/csv"
        )
