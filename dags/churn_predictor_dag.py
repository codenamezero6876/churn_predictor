from airflow.decorators import dag, task
from datetime import datetime

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_preprocessing import process_data_
from feature_engineering import engineer_features_
from model_training import train_model_
from model_evaluation import evaluate_model_


@dag(
    start_date=datetime(year=2026, month=3, day=8, hour=9, minute=0),
    schedule="@daily",
    catchup=True,
    max_active_runs=1
)
def pipeline_flow():
    @task()
    def process_data():
        process_data_()

    @task()
    def engineer_features(raw_data):
        engineer_features_()

    @task()
    def train_model(transformed_data):
        train_model_()

    @task()
    def evaluate_model(transformed_data):
        evaluate_model_()

    chain(process_data(), engineer_features(), train_model(), evaluate_model())

pipeline_flow()
