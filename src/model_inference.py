from pyspark.ml.classification import \
    DecisionTreeClassificationModel, \
    GBTClassificationModel, \
    LinearSVCModel, \
    LogisticRegressionModel, \
    NaiveBayesModel, \
    RandomForestClassificationModel

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Transformer
from pyspark.sql import SparkSession, DataFrame

import os
import json

from src.helper_class import \
    Log, \
    LoadYamlParams, \
    SparkLoader, \
    load_data


logger = Log.setup_logging()


class ModelEvaluator:
    """
    Class for evaluating the trained model and generating performance metrics.
    """

    def __init__(self, spark: SparkSession, config: dict):
        """
        Initialize evaluator with SparkSession.

        Args:
            spark (SparkSession): Existing SparkSession to use
            config (dict): Model evaluation configurations to use
        """
        self.spark = spark
        self.config = config
        logger.info("[INFO] ModelEvaluator initialized with existing SparkSession")

    def load_model(self, model_path: str, model_id: str):
        """Load the trained model from disk."""
        try:
            if model_id.startswith("GBTClassifier"):
                model = GBTClassificationModel.load(model_path)
            elif model_id.startswith("RandomForestClassifier"):
                model = RandomForestClassificationModel.load(model_path)
            elif model_id.startswith("DecisionTreeClassifier"):
                model = DecisionTreeClassificationModel.load(model_path)
            elif model_id.startswith("LogisticRegression"):
                model = LogisticRegressionModel.load(model_path)
            elif model_id.startswith("LinearSVC"):
                model = LinearSVCModel.load(model_path)
            elif model_id.startswith("NaiveBayes"):
                model = NaiveBayesModel.load(model_path)
            else:
                logger.info("[WARNING] Model type is invalid or unavailable. Defaulting to base GBTClassifierModel.")
                model = GBTClassificationModel()
            logger.info(f"[INFO] Model loaded successfully from {model_path}")
            return model
        
        except Exception as e:
            logger.error(f"[ERROR] Error loading model: {str(e)}", stacklevel=2)
            raise

    def load_registered_model(self):
        """Load the trained model from MLflow."""
        try:
            logger.info("[INFO] Loading registered model from MLflow...")
            
            model_uri = self.config.get("uri")
            model_name = self.config.get("registered_model_name")

            mlflow.set_tracking_uri(model_uri)
            reg_models = mlflow.search_registered_models(
                filter_string=f"name = '{model_name}'"
            )

            reg_model_version = reg_models[0].latest_versions[0].version
            reg_model = mlflow.spark.load_model(
                model_uri=f"models:/{model_name}/{reg_model_version}"
            )
            logger.info(f"[INFO] Model version {reg_model_version} loaded from model registry")
            return reg_model

        except Exception as e:
            logger.error(f"[ERROR] Error loading registered model: {str(e)}", stacklevel=2)
            raise

    def load_inference_data(self, input_path: str) -> DataFrame:
        load_format = self.config.get("load_format", "delta")
        df = load_data(
            spark=self.spark,
            input_path=input_path,
            format=load_format
        )
        return df

    def predict(self, model: Transformer, df: DataFrame):
        """Perform prediction on inference data."""
        try:
            logger.info("[INFO] Predicting data...")
            preds = model.transform(df)
            logger.info("[INFO] Prediction completed successfully")
            return json.dumps(preds)
        
        except Exception as e:
            logger.error(f"[ERROR] Error predicting data: {str(e)}", stacklevel=2)
            raise


def predict_churn_():
    """Main function to run model evaluation."""

    logger.info("[INFO] Starting model inference pipeline...")

    try:
        params_obj = LoadYamlParams()
        params = params_obj.load_params(filepath="params.yaml")

        app_name = params["sparksession"]["name"]
        model_id = params["paths"]["model_path"]
        model_path = os.path.join(
            params["paths"]["artifacts"],
            model_id
        )
        inference_path = params["paths"]["data"]["inference"]

        spark_loader = SparkLoader(app_name)
        predictor = ModelPredictor(
            spark=spark_loader.spark,
            config=params["model_evaluation"]
        )
        model = predictor.load_model(model_path, model_id)
        inference_data = predictor.load_inference_data(inference_path)
        predictions = predictor.predict(model, inference_data)

        logger.info("[INFO] Model inference completed successfully")
        return predictions

    except Exception as e:
        logger.error(f"[ERROR] Model inference pipeline failed: {str(e)}")
        raise

    finally:
        spark_loader.spark.stop()
        logger.info("Spark session closed")


if __name__ == "__main__":
    evaluate_model_()
