# Databricks Notebook

<br>
<br>

## Load Delta Table

```python
df = spark.read \
    .format("delta") \
    .load("abfss://visual_container_name@storage_account_name.dfs.core.windows.net/customers")

df.write.format("delta").mode("overwrite").saveAsTable("customer_churn")
```

<br>
<br>

## SQL Queries for Dashboard

### KPI Metrics

```sql
-- Visualize as Single Value / KPI Cards
SELECT 
  COUNT(*) AS total_customers,
  SUM(CASE WHEN Churn = true THEN 1 ELSE 0 END) AS churned_customers,
  ROUND(100.0 * SUM(CASE WHEN Churn = true THEN 1 ELSE 0 END)/COUNT(*), 2) AS churn_rate,
  ROUND(AVG(Tenure), 2) AS avg_tenure,
  ROUND(AVG(Total_Spend), 2) AS avg_spend
FROM customer_churn;
```

<br>

### Churn Distribution

```sql
-- Visualize as Pie Chart
SELECT Churn, COUNT(*) AS count
FROM customer_churn
GROUP BY churn;
```

<br>

### Churn by Subscription Type

```sql
-- Visualize as Stacked Bar Chart
SELECT Subscription_Type, Churn, COUNT(*) AS count
FROM customer_churn
GROUP BY Subscription_Type, Churn;
```

<br>

### Churn by Contract Length

```sql
-- Visualize as Stacked Bar Chart
SELECT Contract_Length, Churn, COUNT(*) AS count
FROM customer_churn
GROUP BY Contract_Length, Churn;
```

<br>

### Churn by Tenure

```sql
-- Visualize as Stacked Bar Chart
SELECT Tenure, Churn, COUNT(*) AS count
FROM customer_churn
GROUP BY Tenure, Churn;
```

<br>

### Churn by Support Call

```sql 
SELECT Support_Calls, Churn, COUNT(*) AS count
FROM customer_churn
GROUP BY Support_Calls, Churn
ORDER BY Support_Calls;
```

<br>

### Average Metrics by Churn

```sql
-- Visualize as Table
SELECT 
  churn,
  AVG(Age) AS avg_age,
  AVG(Tenure) AS avg_tenure,
  AVG(Usage_Frequency) AS avg_usage,
  AVG(Support_Calls) AS avg_support_calls,
  AVG(Payment_Delay) AS avg_payment_delay,
  AVG(Total_Spend) AS avg_spend
FROM customer_churn
GROUP BY Churn;
```

<br>
<br>

## Dashboard Layout

### 🔝 Row 1 (KPIs)
-   Total Customers
-   Churn Rate
-   Avg Tenure
-   Avg Spend

### 📊 Row 2
-   Churn Distribution (Pie)
-   Churn by Subscription Type

### 📊 Row 3
-   Churn by Contract Length
-   Tenure Group vs Churn

### 📉 Row 4
-   Support Calls vs Churn
-   Usage Frequency vs Churn
