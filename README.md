# 🏢 Customer Churn Predictor Pipeline

## 📋 Project Description

This project implements an end-to-end machine learning pipeline to predict customer churn, enabling businesses to proactively identify at-risk customers and take targeted retention actions.

Customer churn is a critical business problem that directly impacts revenue and growth. In this project, I built a complete ML workflow—from data preprocessing to model deployment—to accurately classify customers likely to discontinue a service.

<br>

## 🧩 Key Features
- End-to-End Pipeline: Covers data ingestion, data cleaning, feature engineering, model training, evaluation, and prediction.
- Feature Engineering: Derived meaningful customer behavior indicators such as usage patterns and engagement metrics.
- Modeling & Evaluation: Trained and compared multiple machine learning models to optimize performance using metrics like ROC-AUC, precision, and recall.
- Model Explainability: Integrated feature importance analysis to interpret key drivers of churn.
- Reproducibility: Structured pipeline for consistent and repeatable results.

<br>

## 🛠️ Tech Stack

![My Skills](https://go-skill-icons.vercel.app/api/icons?i=azure,airflow,databricks,delta,mlflow,pyspark,python)

- ADLS Gen2
- Airflow
- Databricks (dashboard to be done soon)
- Delta
- MLflow
- PySpark
- Python


<br>

## 📂 Project Structure

```
│
├── dags/
│   ├── training_pipeline.py        # Training pipeline
│   ├── inference_pipeline.py       # Inference pipeline
│
├── data/
│   ├── raw/                        # Raw churn dataset
│   ├── ingested/                   # Ingested data
│   ├── processed/                  # Transformed churn dataset (ready to be used for ML and visualization)
│   ├── visual/                     # Data to be used for visual analysis
│   ├── train/                      # Training data
│   ├── testing/                    # Testing data
│   ├── inference/                  # Inference data
│
├── logging-info/
│   ├── logs.log                    # Logging to monitor execution and troubleshoot errors
│
├── src/
│   ├── data_ingestion.py           # Get data from cloud storage
│   ├── data_preprocessing.py       # Perform ETL (cleaning, filtering) on ingested data
│   ├── feature_engineering.py      # Perform ETL (transforming into relevant/meaningful features)
│   ├── model_training.py           # Train the models with the transformed data (with hyperparameter tuning)
│   ├── model_inference.py          # Perform model prediction with inference data
│   ├── registry_update.py          # Update MLflow model registry
│   ├── helper_class.py             # Common classes and functions to be used for other scripts
│   ├── data_visualization.py       # Prepares data for visual analysis / dashboard
│   ├── databricks_notebook.md      # Databricks notebook code to create dashboard
│
├── params.yaml                     # Configurations for each stage of the pipeline
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
```

<br>

## 📈 Results

The final model achieved strong predictive performance, effectively identifying high-risk customers while balancing precision and recall. Insights from the model highlight key factors influencing churn, providing actionable business recommendations.

<br>

## 💡 Business Impact

By predicting churn in advance, this solution can help organizations:

-  Improve customer retention strategies
-  Reduce revenue loss
-  Enable targeted marketing interventions
