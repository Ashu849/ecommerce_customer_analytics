from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


st.set_page_config(
    page_title="E-Commerce Analytics",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
sns.set_theme(style="whitegrid")


@st.cache_data
def load_data():
    """Read, merge, and prepare the three project CSV files."""
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    products = pd.read_csv(DATA_DIR / "products.csv")
    orders = pd.read_csv(DATA_DIR / "orders.csv")

    orders = orders.drop_duplicates().copy()
    df = orders.merge(
        customers, on="Customer_ID", how="left", validate="many_to_one"
    )
    df = df.merge(
        products, on="Product_ID", how="left", validate="many_to_one"
    )

    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
    df["Discount_Percent"] = df["Discount_Percent"].fillna(0)
    df["Customer_Rating"] = df["Customer_Rating"].fillna(
        df["Customer_Rating"].median()
    )
    df["Gross_Sales"] = df["Quantity"] * df["Unit_Price"]
    df["Discount_Amount"] = (
        df["Gross_Sales"] * df["Discount_Percent"] / 100
    )
    df["Net_Sales"] = df["Gross_Sales"] - df["Discount_Amount"]
    df["Total_Cost"] = df["Quantity"] * df["Unit_Cost"]
    df["Profit"] = df["Net_Sales"] - df["Total_Cost"]
    df["Profit_Margin"] = np.where(
        df["Net_Sales"] > 0,
        df["Profit"] / df["Net_Sales"] * 100,
        0,
    )
    df["Year_Month"] = df["Order_Date"].dt.to_period("M").astype(str)
    return df


def apply_filters(df):
    """Create sidebar filters and return the filtered data."""
    st.sidebar.header("Dashboard Filters")

    categories = sorted(df["Category"].dropna().unique())
    states = sorted(df["State"].dropna().unique())
    statuses = sorted(df["Order_Status"].dropna().unique())

    selected_categories = st.sidebar.multiselect(
        "Product category", categories, default=categories
    )
    selected_states = st.sidebar.multiselect(
        "Customer state", states, default=states
    )
    selected_statuses = st.sidebar.multiselect(
        "Order status", statuses, default=statuses
    )

    minimum_date = df["Order_Date"].min().date()
    maximum_date = df["Order_Date"].max().date()
    selected_dates = st.sidebar.date_input(
        "Order date range",
        value=(minimum_date, maximum_date),
        min_value=minimum_date,
        max_value=maximum_date,
    )

    filtered = df[
        df["Category"].isin(selected_categories)
        & df["State"].isin(selected_states)
        & df["Order_Status"].isin(selected_statuses)
    ].copy()

    if len(selected_dates) == 2:
        start_date = pd.Timestamp(selected_dates[0])
        end_date = pd.Timestamp(selected_dates[1])
        filtered = filtered[
            filtered["Order_Date"].between(start_date, end_date)
        ]

    return filtered


def show_kpis(df):
    """Display the main business performance indicators."""
    total_sales = df["Net_Sales"].sum()
    total_profit = df["Profit"].sum()
    total_orders = df["Order_ID"].nunique()
    total_customers = df["Customer_ID"].nunique()
    average_order = total_sales / total_orders if total_orders else 0
    margin = total_profit / total_sales * 100 if total_sales else 0

    first, second, third, fourth = st.columns(4)
    first.metric("Total Sales", f"₹{total_sales:,.0f}")
    second.metric("Total Profit", f"₹{total_profit:,.0f}")
    third.metric("Total Orders", f"{total_orders:,}")
    fourth.metric("Customers", f"{total_customers:,}")

    fifth, sixth = st.columns(2)
    fifth.metric("Average Order Value", f"₹{average_order:,.0f}")
    sixth.metric("Profit Margin", f"{margin:.1f}%")


def show_sales_charts(df):
    """Display monthly trend and category performance charts."""
    monthly = (
        df.groupby("Year_Month", as_index=False)[["Net_Sales", "Profit"]]
        .sum()
        .sort_values("Year_Month")
    )
    category = (
        df.groupby("Category", as_index=False)[["Net_Sales", "Profit"]]
        .sum()
        .sort_values("Net_Sales", ascending=False)
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Monthly Sales and Profit")
        figure, axis = plt.subplots(figsize=(8, 5))
        sns.lineplot(
            data=monthly,
            x="Year_Month",
            y="Net_Sales",
            marker="o",
            label="Sales",
            ax=axis,
        )
        sns.lineplot(
            data=monthly,
            x="Year_Month",
            y="Profit",
            marker="o",
            label="Profit",
            ax=axis,
        )
        axis.set_xlabel("Month")
        axis.set_ylabel("Amount (INR)")
        axis.tick_params(axis="x", rotation=45)
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

    with right:
        st.subheader("Sales by Product Category")
        figure, axis = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=category,
            x="Net_Sales",
            y="Category",
            hue="Category",
            palette="viridis",
            legend=False,
            ax=axis,
        )
        axis.set_xlabel("Net Sales (INR)")
        axis.set_ylabel("Category")
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)


def show_profit_charts(df):
    """Display discount impact and correlation charts."""
    left, right = st.columns(2)

    with left:
        st.subheader("Discount Impact on Profit")
        figure, axis = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            data=df,
            x="Discount_Percent",
            y="Profit",
            hue="Category",
            size="Net_Sales",
            ax=axis,
        )
        axis.axhline(0, color="red", linestyle="--", linewidth=1)
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

    with right:
        st.subheader("Correlation Heatmap")
        numeric_columns = [
            "Quantity",
            "Unit_Price",
            "Discount_Percent",
            "Net_Sales",
            "Profit",
            "Customer_Rating",
        ]
        figure, axis = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            df[numeric_columns].corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            ax=axis,
        )
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)


def create_rfm(df):
    """Calculate Recency, Frequency, Monetary value and customer segment."""
    reference_date = df["Order_Date"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("Customer_ID", as_index=False).agg(
        Recency=(
            "Order_Date", lambda value: (reference_date - value.max()).days
        ),
        Frequency=("Order_ID", "nunique"),
        Monetary=("Net_Sales", "sum"),
    )
    rfm["Customer_Segment"] = np.select(
        [
            (rfm["Recency"] <= 30) & (rfm["Frequency"] >= 5),
            (rfm["Recency"] <= 60) & (rfm["Frequency"] >= 3),
            rfm["Recency"] > 90,
        ],
        ["Champion", "Loyal Customer", "At-Risk Customer"],
        default="Regular Customer",
    )
    return rfm.sort_values("Monetary", ascending=False)


def show_customer_analysis(df):
    """Display top-customer and RFM tables."""
    top_customers = (
        df.groupby(["Customer_ID", "Customer_Name"], as_index=False)
        .agg(
            Total_Spent=("Net_Sales", "sum"),
            Orders=("Order_ID", "nunique"),
            Profit=("Profit", "sum"),
        )
        .nlargest(10, "Total_Spent")
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Top 10 Customers")
        st.dataframe(
            top_customers.style.format(
                {"Total_Spent": "₹{:,.0f}", "Profit": "₹{:,.0f}"}
            ),
            use_container_width=True,
        )
    with right:
        st.subheader("RFM Customer Segmentation")
        rfm = create_rfm(df)
        st.dataframe(
            rfm.head(10).style.format({"Monetary": "₹{:,.0f}"}),
            use_container_width=True,
        )


def main():
    st.title("E-Commerce Sales and Customer Analytics")
    st.caption(
        "Interactive analysis using Pandas, NumPy, Matplotlib, Seaborn and Streamlit"
    )

    try:
        complete_data = load_data()
    except FileNotFoundError as error:
        st.error(
            "CSV file not found. Keep customers.csv, products.csv and "
            "orders.csv inside the Data folder next to app.py."
        )
        st.code(str(error))
        st.stop()
    except Exception as error:
        st.error("The project data could not be loaded.")
        st.exception(error)
        st.stop()

    filtered_data = apply_filters(complete_data)
    if filtered_data.empty:
        st.warning("No records match the selected filters.")
        st.stop()

    show_kpis(filtered_data)
    st.divider()
    show_sales_charts(filtered_data)
    st.divider()
    show_profit_charts(filtered_data)
    st.divider()
    show_customer_analysis(filtered_data)

    st.subheader("Filtered Order Data")
    st.dataframe(filtered_data, use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered data as CSV",
        data=filtered_data.to_csv(index=False).encode("utf-8"),
        file_name="filtered_ecommerce_data.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
