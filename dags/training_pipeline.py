from airflow.decorators import dag, task
from datetime import datetime

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data ingestion import ingest_data_
from data_preprocessing import process_data_
from feature_engineering import engineer_features_
from model_training import train_model_
from registry_update import update_registry_


@dag(
    start_date=datetime(year=2026, month=3, day=8, hour=9, minute=0),
    schedule="@daily",
    catchup=True,
    max_active_runs=1
)
def pipeline_flow():
    @task()
    def ingest_data():
        ingest_data_()
    
    @task()
    def process_data():
        process_data_()

    @task()
    def engineer_features():
        engineer_features_()

    @task()
    def train_model():
        train_model_()

    @task()
    def update_registry():
        update_registry_()

    chain(
        ingest_data(),
        process_data(), 
        engineer_features(), 
        train_model(), 
        update_registry()
    )

pipeline_flow()
