from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml import Transformer
from pyspark.sql import SparkSession, DataFrame

import os
import json

from src.data_preprocessing import Log, LoadYamlParams, SparkLoader


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
        logger.info("ModelEvaluator initialized with existing SparkSession")

    def load_model(self, model_path: str):
        """Load the trained model from disk."""
        try:
            model = RandomForestClassificationModel.load(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
            return model
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def load_data_parquet(self, input_path: str) -> DataFrame:
        """Load test data from local Parquet files."""

        try:
            df = self.spark.read.parquet(input_path)
            logger.info(f"Test data successfully loaded from {input_path}")
            return df

        except Exception as e:
            logger.error(f"Error loading files: {str(e)}")
            raise

    def calculate_metrics(self, model: Transformer, df: DataFrame):
        """Calculate various classification metrics."""
        try:
            preds = model.transform(df)

            binary_evaluator = BinaryClassificationEvaluator(
                rawPredictionCol="rawPrediction", labelCol="label"
            )
            multi_evaluator = MulticlassClassificationEvaluator(
                predictionCol="prediction", labelCol="label"
            )

            metrics = {
                "accuracy": multi_evaluator.setMetricName("accuracy").evaluate(preds),
                "precision": multi_evaluator.setMetricName("weightedPrecision").evaluate(preds),
                "recall": multi_evaluator.setMetricName("weightedRecall").evaluate(preds),
                "f1": multi_evaluator.setMetricName("f1").evaluate(preds),
                "auc_roc": binary_evaluator.setMetricName("areaUnderROC").evaluate(preds)
            }

            cf_matrix = preds.groupBy("label", "prediction").count().collect()
            metrics["confusion_matrix"] = [
                {
                    "actual": row["label"],
                    "predicted": row["prediction"],
                    "count": row["count"]
                }
                for row in cf_matrix
            ]

            logger.info("Model evaluation metrics calculated successfully")
            return metrics

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise

    def save_metrics(self, metrics: dict, metrics_path: str):
        """Save metrics to JSON file."""
        try:
            output_path = os.path.join(metrics_path, "model_metrics.json")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(metrics, f, indent=4)

            logger.info(f"Metrics saved successfully to {output_path}")

        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
            raise



def main():
    """Main function to run model evaluation."""
    try:
        params_obj = LoadYamlParams()
        params = params_obj.load_params(filepath="params.yaml")

        app_name = params["sparksession"]["name"]
        model_path = os.path.join(
            params["paths"]["artifacts_path"],
            params["model_training"]["model_name"]
        )
        test_path = params["paths"]["test_data_path"]
        metrics_path = params["paths"]["metrics_path"]

        spark_loader = SparkLoader(app_name)
        evaluator = ModelEvaluator(
            spark=spark_loader.spark,
            config=params["model_evaluation"]
        )
        model = evaluator.load_model(model_path)
        test_data = evaluator.load_data_parquet(test_path)
        metrics = evaluator.calculate_metrics(model, test_data)
        evaluator.save_metrics(metrics, metrics_path)

        logger.info("Model evaluation completed successfully")

    except Exception as e:
        logger.error(f"Model evaluation pipeline failed: {str(e)}")
        raise

    finally:
        spark_loader.spark.stop()
        logger.info("Spark session closed")


if __name__ == "__main__":
    main()