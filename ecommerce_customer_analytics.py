from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
CHARTS_DIR = BASE_DIR / "charts"
OUTPUTS_DIR = BASE_DIR / "outputs"
CHARTS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", palette="deep")


def load_and_clean_data():
    """Load, merge, and clean the three project datasets."""
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    products = pd.read_csv(DATA_DIR / "products.csv")
    orders = pd.read_csv(DATA_DIR / "orders.csv")

    duplicate_rows = orders.duplicated().sum()
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
    return df, duplicate_rows


def add_calculated_columns(df):
    """Calculate sales, cost, profit, time, and order categories."""
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
    df["Order_Size"] = np.select(
        [
            df["Net_Sales"] < 1000,
            df["Net_Sales"].between(1000, 5000),
            df["Net_Sales"] > 5000,
        ],
        ["Small", "Medium", "Large"],
        default="Unknown",
    )
    df["Customer_Type"] = np.where(
        df["Net_Sales"] >= df["Net_Sales"].median(),
        "High-Value",
        "Regular",
    )
    df["Year"] = df["Order_Date"].dt.year
    df["Month"] = df["Order_Date"].dt.month_name()
    df["Year_Month"] = df["Order_Date"].dt.to_period("M").astype(str)
    return df


def print_kpis(df, duplicate_rows):
    """Display important business KPIs."""
    total_sales = df["Net_Sales"].sum()
    total_profit = df["Profit"].sum()
    total_orders = df["Order_ID"].nunique()
    total_customers = df["Customer_ID"].nunique()
    average_order_value = total_sales / total_orders if total_orders else 0
    profit_margin = total_profit / total_sales * 100 if total_sales else 0

    print("\nE-COMMERCE ANALYSIS SUMMARY")
    print(f"Duplicate rows removed: {duplicate_rows}")
    print(f"Total Sales: INR {total_sales:,.2f}")
    print(f"Total Profit: INR {total_profit:,.2f}")
    print(f"Total Orders: {total_orders:,}")
    print(f"Total Customers: {total_customers:,}")
    print(f"Average Order Value: INR {average_order_value:,.2f}")
    print(f"Profit Margin: {profit_margin:.2f}%")


def create_summary_tables(df):
    """Create category, customer, outlier, and RFM tables."""
    category_analysis = (
        df.groupby("Category", as_index=False)
        .agg(
            Total_Sales=("Net_Sales", "sum"),
            Total_Profit=("Profit", "sum"),
            Total_Orders=("Order_ID", "nunique"),
        )
        .sort_values("Total_Sales", ascending=False)
    )
    top_customers = (
        df.groupby(["Customer_ID", "Customer_Name"], as_index=False)
        .agg(
            Total_Spent=("Net_Sales", "sum"),
            Total_Orders=("Order_ID", "nunique"),
            Total_Profit=("Profit", "sum"),
        )
        .nlargest(10, "Total_Spent")
    )

    q1 = np.percentile(df["Net_Sales"], 25)
    q3 = np.percentile(df["Net_Sales"], 75)
    iqr = q3 - q1
    outliers = df[
        (df["Net_Sales"] < q1 - 1.5 * iqr)
        | (df["Net_Sales"] > q3 + 1.5 * iqr)
    ].copy()

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

    category_analysis.to_csv(
        OUTPUTS_DIR / "category_analysis.csv", index=False
    )
    top_customers.to_csv(OUTPUTS_DIR / "top_10_customers.csv", index=False)
    outliers.to_csv(OUTPUTS_DIR / "unusual_orders.csv", index=False)
    rfm.to_csv(OUTPUTS_DIR / "rfm_customer_analysis.csv", index=False)
    return category_analysis


def save_chart(file_name):
    """Save, display, and close the active chart."""
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / file_name, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close()


def create_charts(df, category_analysis):
    """Create four business-analysis charts."""
    monthly_sales = (
        df.groupby("Year_Month", as_index=False)[["Net_Sales", "Profit"]]
        .sum()
    )

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=monthly_sales,
        x="Year_Month",
        y="Net_Sales",
        marker="o",
        label="Sales",
    )
    sns.lineplot(
        data=monthly_sales,
        x="Year_Month",
        y="Profit",
        marker="o",
        label="Profit",
    )
    plt.title("Monthly Sales and Profit Trend")
    plt.xlabel("Month")
    plt.ylabel("Amount (INR)")
    plt.xticks(rotation=45)
    save_chart("monthly_sales_profit_trend.png")

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=category_analysis,
        x="Total_Sales",
        y="Category",
        hue="Category",
        palette="viridis",
        legend=False,
    )
    plt.title("Sales by Product Category")
    plt.xlabel("Total Sales (INR)")
    save_chart("sales_by_category.png")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="Discount_Percent",
        y="Profit",
        hue="Category",
        size="Net_Sales",
    )
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Impact of Discount on Profit")
    save_chart("discount_vs_profit.png")

    numeric_columns = [
        "Quantity",
        "Unit_Price",
        "Discount_Percent",
        "Net_Sales",
        "Profit",
        "Customer_Rating",
    ]
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        df[numeric_columns].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
    )
    plt.title("Sales Data Correlation")
    save_chart("correlation_heatmap.png")


def main():
    df, duplicate_rows = load_and_clean_data()
    df = add_calculated_columns(df)

    print("First five merged records:")
    print(df.head())
    print(f"\nData shape after cleaning: {df.shape}")
    print_kpis(df, duplicate_rows)

    df.to_csv(DATA_DIR / "cleaned_sales_data.csv", index=False)
    category_analysis = create_summary_tables(df)
    create_charts(df, category_analysis)

    print("\nProject completed successfully.")
    print(f"Cleaned data saved in: {DATA_DIR}")
    print(f"Charts saved in: {CHARTS_DIR}")
    print(f"Summary files saved in: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
