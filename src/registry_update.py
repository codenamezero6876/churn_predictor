from pyspark.ml import Model

from pyspark.sql import SparkSession, DataFrame
import mlflow
import os, yaml
from src.helper_class import \
    Log, \
    SparkLoader, \
    LoadYamlParams, \
    load_data


logger = Log.setup_logging()


class RegistryUpdater:
    """
    Updates the model registry by comparing model performances
    """

    def __init__(self, crit_val: float, metric: str, config: dict):
        self.crit_val = crit_val
        self.metric = metric
        self.config = config

    def update(self):
        try:
            logger.info("[INFO] Updating model registry...")

            mlflow.set_tracking_uri(self.config.get("uri"))
            run_id = self.config.get("mlflow_run_id")

            model_run = mlflow.get_run(run_id=run_id)
            model_metric = model_run.data.metrics.get(self.metric)

            reg_models = mlflow.search_registered_models(
                filter_string=f"name = '{self.config.get("registered_model_name")}'"
            )

            if not reg_models:
                mlflow.register_model(
                    model_uri=f"runs:/{run_id}/{self.config.get("artifact_path")}",
                    name=self.config.get("registered_model_name")
                )
                logger.info(f"[INFO] New model registered")

            reg_run_id = reg_models[0].latest_versions[0].run_id
            reg_model_run = mlflow.get_run(run_id=reg_run_id)
            reg_model_metric = reg_model_run.data.metrics.get(self.metric)

            if model_metric > reg_model_metric * (1 + self.crit_val):
                mlflow.register_model(
                    model_uri=f"runs:/{run_id}/{self.config.get("artifact_path")}",
                    name=self.config.get("registered_model_name")
                )
                logger.info(f"[INFO] Model registry updated")

        except Exception as e:
            logger.error(f"[ERROR] Error updating model registry: {str(e)}", stacklevel=2)
            raise

def update_registry_():
    """Main function to update model registry."""

    try:
        logger.info("[INFO] Starting Registry Update Pipeline...")

        params_obj = LoadYamlParams()
        params = params_obj.load_params(filepath="params.yaml")

        crit_val = params["mlflow"]["registry"]["crit_val"]
        metric = params["mlflow"]["registry"]["metric"]

        updater = RegistryUpdater(
            crit_val=crit_val,
            metric=metric,
            config=params["mlflow"]
        )
        updater.update()
        logger.info("[INFO] Registry Update Pipeline completed successfully")

    except Exception as e:
        logger.error(f"[ERROR] Registry Update Pipeline Failed: {str(e)}", stacklevel=2)
        raise
