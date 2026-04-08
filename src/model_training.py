from pyspark.ml.classification import \
    DecisionTreeClassifier, \
    GBTClassifier, \
    LinearSVC, \
    LogisticRegression, \
    NaiveBayes, \
    RandomForestClassifier

from pyspark.ml.tuning import \
    ParamGridBuilder, \
    CrossValidator

from pyspark.ml.evaluation import \
    BinaryClassificationEvaluator, \
    MulticlassClassificationEvaluator

from pyspark.sql import SparkSession, DataFrame

import optuna
import os, yaml
from src.helper_class import \
    Log, \
    SparkLoader, \
    LoadYamlParams, \
    load_data


logger = Log.setup_logging()


### TO DO: Create config for param_grid in .yml file
class ModelTrainer:
    """
    This class handles model training operations.
    """

    def __init__(self, spark: SparkSession, config: dict):
        """
        Initialize model trainer with SparkSession.

        Args:
            spark (SparkSession): Existing SparkSession to use
            config (dict): Model training configurations to use
        """
        self.spark = spark
        self.config = config
        logger.info("[INFO] ModelTrainer initialized with existing SparkSession")

    def load_train_data(self, input_path: str) -> DataFrame:
        load_format = self.config.get("load_format", "delta")
        """Load training data from local Parquet files."""

        df = load_data(
            spark=self.spark,
            input_path=input_path,
            format=load_format
        )
        return df

    def _build_param_grid(self, model_classifier, grid_values:dict):
        """Builds a parameter grid as the search space for optimal parameters."""
        try:
            logger.info("[INFO] Building parameter grid for the classifier...")
            param_grid = ParamGridBuilder()
            for param_name, values in grid_values.items():
                param_grid.addGrid(model_classifier.getParam(param_name), values)
            
            logger.info("[INFO] Parameter grid built successfully")
            return param_grid.build()

        except Exception as e:
            logger.error(f"[ERROR] Error building parameter grid: {str(e)}", stacklevel=2)
            raise

    def train(self, train_data: DataFrame):
        """
        Train a classifier model.

        Args:
            train_data (DataFrame): Data to train the model

        Returns:
            The trained model
        """
        try:
            model_type = self.config["model_choice"]
            optim_type = self.config["optim_choice"]
            evaluator_type = self.config["eval_choice"]

            if model_type == "gbt":
                model_classifier = GBTClassifier()
            elif model_type == "rfc":
                model_classifier = RandomForestClassifier()
            elif model_type == "dtc":
                model_classifier = DecisionTreeClassifier()
            elif model_type == "lr":
                model_classifier = LogisticRegression()
            elif model_type == "svc":
                model_classifier = LinearSVC()
            elif model_type == "nb":
                model_classifier = NaiveBayes()
            else:
                logger.info("[WARNING] Model type is invalid or unavailable. Defaulting to gbt.")
                optim_type = None
                model_classifier = GBTClassifier()
            
            if optim_type == "grid":
                grid_values = self.config["optimization"]["grid"][model_type]
                param_grid = self._build_param_grid(model_classifier, grid_values)

                if evaluator_type == "areaUnderROC" or evaluator_type == "areaUnderPR":
                    evaluator = BinaryClassificationEvaluator(metricName=evaluator_type)
                else:
                    evaluator = MulticlassClassificationEvaluator(metricName=evaluator_type)

                model_cv = CrossValidator(
                    estimator=model_classifier,
                    evaluator=evaluator,
                    estimatorParamMaps=param_grid
                )
                logger.info("[INFO] Starting model training...")
                best_model = model_cv.fit(train_data).bestModel
                logger.info("[INFO] Model training completed successfully")
                return best_model
            
            else:
                logger.info("[INFO] Starting model training...")
                model = model_classifier.fit(train_data)
                logger.info("[INFO] Model training completed successfully")
                return model

        except Exception as e:
            logger.error(f"[ERROR] Error training model: {str(e)}", stacklevel=2)
            raise

    def evaluate_model(self, model: Model, test_data: DataFrame):
        """
        Evaluates the (best) trained model returned from train().
        """
        try:
            logger.info(f"[INFO] Evaluating model performance...")

            param_map = {
                k.name: v
                for k, v in model.extractParamMap().items()
                if k.name != "doc"
            }

            acc_eval = MulticlassClassificationEvaluator(metricName='accuracy')
            prec_eval = MulticlassClassificationEvaluator(metricName='weightedPrecision')
            recall_eval = MulticlassClassificationEvaluator(metricName='weightedRecall')
            f1_eval = MulticlassClassificationEvaluator(metricName='f1')
            roc_auc_eval = BinaryClassificationEvaluator(metricName='areaUnderROC')
            area_under_pr_eval = BinaryClassificationEvaluator(metricName='areaUnderPR')

            results = model.transform(test_data)
            metrics = {
                "accuracy": acc_eval.evaluate(results),
                "weightedPrecision": prec_eval.evaluate(results),
                "weightedRecall": recall_eval.evaluate(results),
                "f1": f1_eval.evaluate(results),
                "areaUnderROC": roc_auc_eval.evaluate(results),
                "areaUnderPR": area_under_pr_eval.evaluate(results),
            }
            return param_map, metrics
            
        except Exception as e:
            logger.error(f"[ERROR] Error evaluating model performance: {str(e)}", stacklevel=2)
            raise

    def save_model(self, model, models_path: str, local: bool=False):
        """Save the trained model and returns the model's ID."""
        try:
            if local:
                os.makedirs(os.path.dirname(models_path), exist_ok=True)
            model_path = os.path.join(models_path, model.uid)
            model.save(model_path)
            logger.info(f"[INFO] Model saved successfully to {model_path}")
            return model.uid

        except Exception as e:
            logger.error(f"[ERROR] Error saving trained model: {str(e)}", stacklevel=2)
            raise


def train_model_():
    """Main function to run the model training pipeline."""

    try:
        logger.info("[INFO] Starting model training pipeline...")

        params_obj = LoadYamlParams()
        params = params_obj.load_params(filepath="params.yaml")

        app_name = params["sparksession"]["name"]
        train_path = params["paths"]["data"]["training"]
        test_path = params["paths"]["data"]["testing"]
        models_path = params["paths"]["artifacts"]
        mlflow_uri = params["mlflow"]["uri"]
        mlflow_experiment_name = params["mlflow"]["experiment_name"]
        mlflow_artifact_path = params["mlflow"]["artifact_path"]

        spark_loader = SparkLoader(app_name)
        trainer = ModelTrainer(
            spark=spark_loader.spark,
            config=params["model_training"]
        )

        mlflow_run_id = None
        mlflow.set_tracking_uri(uri=mlflow_uri)
        mlflow.set_experiment(experiment_name=mlflow_experiment_name)

        with mlflow.start_run():
            train_data = trainer.load_train_data(train_path)
            test_data = trainer.load_train_data(test_path)
            trained_model = trainer.train(train_data=train_data)
            param_map, metrics = trainer.evaluate_model(trained_model, test_data)

            mlflow.log_params(param_map)
            mlflow.log_metrics(metrics)
            mlflow.set_tag(key="model_name", value=trained_model.uid)
            mlflow.spark.log_model(
                spark_model=trained_model,
                artifact_path=mlflow_artifact_path
            )
            mlflow_run_id = mlflow.active_run().info.run_id

        model_id = trainer.save_model(trained_model, models_path)
        
        params["paths"]["model_path"] = model_id
        params["mlflow"]["mlflow_run_id"] = mlflow_run_id
        with open("params.yaml", 'w') as f:
            yaml.safe_dump(params, f)

        logger.info("[INFO] Model training pipeline completed successfully")

    except Exception as e:
        logger.error(f"[ERROR] Model training pipeline failed: {str(e)}", stacklevel=2)
        raise

    finally:
        spark_loader.spark.stop()
        logger.info("Spark session closed")


if __name__ == "__main__":
    train_model_()
