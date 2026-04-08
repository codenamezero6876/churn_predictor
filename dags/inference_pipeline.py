from airflow.decorators import dag, task, chain
from datetime import datetime

from ..src.data_ingestion import ingest_data_
from ..src.data_preprocessing import process_data_
from ..src.feature_engineering import engineer_features_
from ..src.model_inference import predict_churn_


@dag(
    start_date=datetime(year=2026, month=3, day=8, hour=9, minute=0),
    schedule="@daily",
    catchup=True,
    max_active_runs=1
)
def pipeline_flow():
    @task()
    def ingest_data():
        ingest_data_(inference_mode=True)

    @task()
    def process_data():
        process_data_(inference_mode=True)

    @task()
    def engineer_features():
        engineer_features_(inference_mode=True)

    @task()
    def predict_churn():
        predict_churn_()


    chain(
        ingest_data(), 
        process_data(), 
        engineer_features(), 
        predict_churn()
    )

pipeline_flow()
