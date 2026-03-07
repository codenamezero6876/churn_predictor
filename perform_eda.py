from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import *
from pyspark.sql.functions import col, count, avg, when, isnan

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Tuple

### Data Ingestion ###
def create_spark_session():
    spark = SparkSession.builder \
        .appName("Dataset EDA") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.parquet.enableVectorizedRender", "true") \
        .config("spark.sql.parquet.columnarReaderBatchSize", "4096") \
        .getOrCreate()
    return spark

def load_data_csv(spark: SparkSession, filepath: str):
    df = spark.read.csv(
        path=filepath,
        header=True,
        inferSchema=True,
        mode="PERMISSIVE"
    )
    return df

def quick_summary(df: DataFrame, num_rows: int=2):
    print(f"Total data count: {df.count()}")
    df.show(num_rows)
    df.describe().show()
    df.printSchema()


### Data preprocessing ###
def analyze_null_values(df: DataFrame) -> None:
    total_count = float(df.count())
    column_types_dict = dict(df.dtypes)

    null_counts = []
    for column in df.columns:
        col_type = column_types_dict[column]

        if col_type in ("int", "double", "float", "long"):
            null_condition = col(column).isNull()
            
            if col_type != "int":
                null_condition = null_condition | isnan(col(column))

        else:
            null_condition = (
                col(column).isNull() |
                (col(column) == "") |
                (col(column) == "NULL") |
                (col(column) == "null")
            )

        null_count = df.filter(null_condition).count()
        percentage = (null_count / total_count) * 100
        null_counts.append((column, null_count, percentage))

    print("\nNull Value Analysis:")
    print("-" * 60)
    print(f"{'Column':<30} {'Null Count':<15} {'Null Percentage'}")
    print("-" * 60)
    for col_name, null_count, percentage in null_counts:
        print(f"{col_name:<30} {null_count:<15} {percentage:.2f}%")

def rate_by_category(df: DataFrame, val_col: str, categories: list[str]):
    print(f"\n{val_col} Rate by Categories:")
    print("-" * 60)

    for cat_col in categories:
        print(f"\nRate by {cat_col}:")
        df.groupBy(cat_col) \
            .agg(
                avg(val_col).alias(f"{val_col}_rate"),
                count("*").alias("count")
            ) \
            .orderBy(f"{val_col}_rate", ascending=False) \
            .show()
        
def calc_correlations(df: DataFrame, val_col: str, numeric_columns: list[str]):
    print(f"Correlations with {val_col}:")
    print("-" * 60)

    correlations = []

    for column in numeric_columns:
        correlation = abs(df.stat.corr(column, val_col))
        correlations.append((column, correlation))

    correlations.sort(key=lambda x: x[1], reverse=True)
    for feature, correlation in correlations:
        print(f"{feature:<20}: {correlation:>10.4f}")

      
### Data Visualization ###
sns.set_theme()
sns.set_palette("husl")

def make_plots_wrt_label(
    df: pd.DataFrame,
    label: str,
    numeric_features: List[str],
    categorical_features: List[str],
    figsize: Tuple[int, int] | None=None,
    suptitle: str="",
    num_cols: int=2,
):
    num_rows = (len(numeric_features) + len(categorical_features) + 1) // 2

    fig = plt.figure(figsize=figsize)

    if suptitle:
        fig.suptitle(suptitle)

    for idx, feature in enumerate(numeric_features, 1):
        plt.subplot(num_rows, num_cols, idx)
        sns.boxplot(x=label, y=feature, data=df)
        plt.title(f"{feature} vs {label}")
        plt.xlabel(f"{label}")

    for idx, feature in enumerate(categorical_features, len(numeric_features) + 1):
        plt.subplot(num_rows, num_cols, idx)
        label_rates = df.groupby(feature)[label].mean()
        label_rates.plot(kind="bar")
        plt.title(f"{label} rate by {feature}")
        plt.ylabel(f"{label} rate")

    plt.tight_layout()
    plt.show()

def make_correlation_heatmap(
    df: pd.DataFrame,
    features: List[str],
    figsize: Tuple[int, int] | None=None,
):
    plt.figure(figsize=figsize)
    corr_mtx = df[features].corr()
    sns.heatmap(corr_mtx, annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

def make_violin_plots_wrt_label(
    df: pd.DataFrame,
    label: str,
    features: List[str],
    figsize: Tuple[int, int] | None=None,
    num_cols: int=2,
):
    plt.figure(figsize=figsize)
    num_rows = (len(features) + 1) // num_cols

    for idx, feature in enumerate(features, 1):
        plt.subplot(num_rows, num_cols, idx)
        sns.violinplot(x=label, y=feature, data=df)
        plt.title(f"{feature} Distribution by {label}")

    plt.tight_layout()
    plt.show()


### Feature Engineering ###

def categorize_age(df: DataFrame):
    df = df.withColumn(
        "age_category",
        when(col("age") < 40, "Young")
        .when((col("age") >= 40) & (col("age") < 65), "Middle-Aged")
        .otherwise("Senior")
    )
    return df

def categorize_balance(df: DataFrame):
    df = df.withColumn(
        "balance_category",
        when(col("balance") == 0, "Zero")
        .when(col("balance") < 50000, "Low") 
        .when(col("balance") < 100000, "Medium")
        .otherwise("High")
    )
    return df

def categorize_credit_score(df: DataFrame):
    df = df.withColumn(
        "credit_score_category",
        when(col("credit_score") < 600, "Poor")
        .when((col("credit_score") >= 600) & (col("credit_score") < 700), "Fair")
        .when((col("credit_score") >= 700) & (col("credit_score") < 800), "Good")
        .otherwise("Excellent")
    )
    return df

def categorize_tenure(df: DataFrame):
    df = df.withColumn(
        "tenure_category",
        when(col("tenure") <= 3, "New")
        .when((col("tenure") > 3) & (col("tenure") <= 12), "Early-Stage")
        .when((col("tenure") > 12) & (col("tenure") <= 24), "Mid-term")
        .otherwise("Loyal")
    )
    return df

def calc_col_to_col_ratio(df: DataFrame, col1: str, col2: str):
    df = df.withColumn(
        f"{col1}_{col2}_ratio",
        col(col1) / when(col(col2) == 0, 1).otherwise(col(col2))
    )

def add_feature_product_engagement_score(df: DataFrame):
    """
    The Product Engagement Score (PES) is calculated as an equal average of 3 key metrics - 
    Adoption, Stickiness and Growth.

    One should measure their PES separately for different user segments. Each segment may 
    have very different engagement metrics, so measuring just an average PES for an entire 
    product can muddle your view of what's going on between different user segments.

    Adoption is a measure of how much of the product features active users are actually using.
    The adoption rate in PES is calculated by taking the average number of features adopted by 
    active users divided by the total number of features available to them.

    Product stickinesss is evaluated by assessing the rate at which users return to the 
    platform. It is important to define what constitutes an active engagement with the product 
    to obtain an accurate stickiness score. To measure stickiness, take the number of Daily 
    Active Users (DAU) in a specific user segment, and divide it by the number of Weekly Active 
    Users (WAU). Alternatively, take the number of WAUs and divide it by the number of Monthly 
    Active Users (MAU).

    Growth measures the increase in paid usage of a product over a given period. It is evaluated 
    using a custom-configured version of the quick ratio, which provides a score between 0 and 
    100 to indicate the net growth of visitors or accounts over a given period. This score is 
    calculated by summing new and recovered accounts and dividing it by churned accounts.
    """
    df = df.withColumn(
        "product_engagement_score",
        col("adoption") * col("stickiness") * col("growth")
    )
    return df


### EDA experiments ###
raw_data_path = "data/raw/customer_churn_dataset-training-master.csv"
spark = create_spark_session()
df = load_data_csv(spark, raw_data_path)

numerical_columns = ["Age", "Tenure", "Usage Frequency", "Support Calls",
                     "Payment Delay", "Total Spend", "Last Interaction"]

categorical_columns = ["Gender", "Subscription Type", "Contract Length"]

# quick_summary(df)

# rate_by_category(df, "Churn", categorical_columns)
# --- FINDINGS ---
#   -   Users with Monthly Contract Length has 1.0 churn rate while those with Annual or Quarterly
#       contract length has 0.46 churn rate.
#   -   Higher churn rate (0.67) for females than males (0.49)
#   -   Similar churn rates (0.56 - 0.58) across all subscription types

# calc_correlations(df, "Churn", numerical_columns)
# --- FINDINGS ---
#   -   Support Calls   : 0.5743
#   -   Total Spend     : 0.4293
#   -   Payment Delay   : 0.3121
#   -   Age             : 0.2184
#   -   Last Interaction: 0.1496
#   -   Tenure          : 0.0519
#   -   Usage Frequency : 0.0461

df_sample = df.sample(0.01, 42)
# df_pandas = df_sample.toPandas()
# make_plots_wrt_label(
#     df=df_pandas,
#     label="Churn",
#     numeric_features=numerical_columns,
#     categorical_features=categorical_columns,
#     suptitle="Relationship between features and Churn outcome"
# )
# make_correlation_heatmap(
#     df=df_pandas,
#     features=numerical_columns + ["Churn"]
# )
# make_violin_plots_wrt_label(
#     df=df_pandas,
#     label="Churn",
#     features=["Support Calls", "Total Spend", "Payment Delay"]
# )
# --- VISUALIZE SMALL SAMPLE OF DATA ---


### To do: Check for skewed data

from functools import reduce
from pyspark.sql import Column
from pyspark.sql import functions as F

dct = {
    "col_name": "Age",
    "labels": [
        {"name": "Young", "start": "null", "end": 40},
        {"name": "Middle-Aged", "start": 40, "end": 65},
        {"name": "Senior", "start": 65, "end": "null"},
    ]
}

def make_connections(category: str, c: Column, label: dict):
    if label["start"] == "null":
        return c.when(col(category) < label["end"], label["name"])
    elif label["end"] == "null":
        return c.when(col(category) > label["start"], label["name"])
    else:
        return c.when(col(category).between(label["start"], label["end"]), label["name"])
    
category = dct["col_name"]
labels = dct["labels"]
    
cond = reduce(
    lambda c, label: make_connections(category, c, label),
    labels,
    F
)

df_sample_modified = df_sample.withColumn(f"{dct['col_name']}_category", cond)

df_sample_modified.select("age_category").printSchema()

spark.stop()