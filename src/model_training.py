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

import os
from src.data_preprocessing import Log, SparkLoader, LoadYamlParams


logger = Log.setup_logging()


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
        logger.info("ModelTrainer initialized with existing SparkSession", stacklevel=2)

    def load_data_parquet(self, input_path: str) -> DataFrame:
        """Load training data from local Parquet files."""

        try:
            df = self.spark.read.parquet(input_path)
            logger.info(f"Training data successfully loaded from {input_path}")
            return df

        except Exception as e:
            logger.error(f"Error loading files: {str(e)}", stacklevel=2)
            raise

    def _build_param_grid(self, model_classifier, grid_values:dict):
        """Builds a parameter grid as the search space for optimal parameters."""
        try:
            logger.info("Building parameter grid for the classifier...")
            param_grid = ParamGridBuilder()
            for param_name, values in grid_values.items():
                param_grid.addGrid(model_classifier.getParam(param_name), values)
            param_grid.build()
            logger.info("Parameter grid built successfully")
            return param_grid

        except Exception as e:
            logger.error(f"Error building parameter grid: {str(e)}", stacklevel=2)
            raise

    def train(
        self,
        train_data: DataFrame,
        model_type: str = "gbt",
        param_optimization: str = None,
        evaluator_type: str = "accuracy"
    ):
        """
        Train a classifier model.

        Args:
            train_data (DataFrame): Data to train the model
            model_type (str): The type of model classifier
            param_optimization (str): Search method for optimal hyperparameters
            evaluator_type (str): Metric type to evaluate during cross validation

        Returns:
            The trained model
        """
        try:
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
                param_optimization = None
                model_classifier = GBTClassifier()
            
            if param_optimization == "grid":
                grid_values = self.config["optimization"]["grid"][model_type]
                param_grid = self._build_param_grid(model_classifier, grid_values)

                if evaluator_type == "areaUnderROC" or evaluator_type == "areaUnderPR":
                    evaluator = BinaryClassificationEvaluator(metricName=evaluator_type)
                else:
                    evaluator = MulticlassClassificationEvaluator(metricName=evaluator_type)

                model_cv = CrossValidator(
                    estimator=model,
                    evaluator=evaluator,
                    estimatorParamMaps=param_grid
                )
                logger.info("Starting model training...")
                best_model = model_cv.fit(train_data).bestModel
                logger.info("Model training completed successfully")
                return best_model
            
            else:
                logger.info("Starting model training...")
                model = model_classifier.fit(train_data)
                logger.info("Model training completed successfully")
                return model

        except Exception as e:
            logger.error(f"Error training model: {str(e)}", stacklevel=2)
            raise

    def save_model(self, model, models_path: str):
        """Save the trained model."""
        try:
            os.makedirs(os.path.dirname(models_path), exist_ok=True)
            model_name = model.uid
            model_path = os.path.join(models_path, model_name)
            model.save(model_path)
            logger.info(f"Model saved successfully to {model_path}")

        except Exception as e:
            logger.error(f"Error saving trained model: {str(e)}", stacklevel=2)
            raise


def train_model_():
    """Main function to run the model training pipeline."""
    try:
        logger.info("Starting model training pipeline...")

        params_obj = LoadYamlParams()
        params = params_obj.load_params(filepath="params.yaml")

        app_name = params["sparksession"]["name"]
        train_path = params["paths"]["train_data_path"]
        models_path = params["paths"]["artifacts_path"]

        spark_loader = SparkLoader(app_name)
        trainer = ModelTrainer(
            spark=spark_loader.spark,
            config=params["model_training"]
        )
        train_data = trainer.load_data_parquet(train_path)
        rf_model = trainer.train_random_forest(train_data)
        trainer.save_model(rf_model, models_path)

        logger.info("Model training pipeline completed successfully")

    except Exception as e:
        logger.error(f"Model training pipeline failed: {str(e)}", stacklevel=2)
        raise

    finally:
        spark_loader.spark.stop()
        logger.info("Spark session closed")


if __name__ == "__main__":
    train_model_()
