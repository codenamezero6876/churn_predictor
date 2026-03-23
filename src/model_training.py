from pyspark.ml.classification import \
    RandomForestClassifier, \
    GBTClassifier, \
    DecisionTreeClassifier, \
    LinearSVC, \
    NaiveBayes
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

    def train_gbt(self, train_data: DataFrame, eval_type="accuracy"):
        """
        Train GBT model with specified parameters.
        """
        try:
            gbt = GBTClassifier(
                labelCol="label",
                featuresCol="features"
            )

            if eval_type == "None":
                logger.info("Starting GBT training...")
                model = gbt.fit(train_data)
                logger.info("GBT training completed successfully")
                return model
            
            param_grid = ParamGridBuilder() \
                .addGrid(gbt.maxDepth, [5, 10, 20]) \
                .addGrid(gbt.featureSubsetStrategy, ["auto", "sqrt", "log2"]) \
                .addGrid(gbt.impurity, [True, False]) \
                .build()
            
            if eval_type == "areaUnderROC" or eval_type == "areaUnderPR":
                evaluator = BinaryClassificationEvaluator(metricName=eval_type)
            else:
                evaluator = MulticlassClassificationEvaluator(metricName=eval_type)

            gbt_cv = CrossValidator(
                estimator=gbt,
                evaluator=evaluator,
                estimatorParamMaps=param_grid
            )

            model = gbt_cv.fit(train_data).bestModel
            return model

        except Exception as e:
            logger.error(f"Error training GBT model: {str(e)}", stacklevel=2)
            raise

    def train_random_forest(self, train_data: DataFrame, eval_type="accuracy"):
        """
        Train Random Forest model with specified parameters.
        """
        try:
            rf = RandomForestClassifier(
                labelCol="label",
                featuresCol="features"
            )

            if eval_type == "None":
                logger.info("Starting Random Forest training...")
                model = rf.fit(train_data)
                logger.info("Random Forest training completed successfully")
                return model

            param_grid = ParamGridBuilder() \
                .addGrid(rf.numTrees, [20, 50, 100]) \
                .addGrid(rf.maxDepth, [5, 10, 20]) \
                .addGrid(rf.featureSubsetStrategy, ["auto", "sqrt", "log2"]) \
                .addGrid(rf.impurity, [True, False]) \
                .build()
            
            if eval_type == "areaUnderROC" or eval_type == "areaUnderPR":
                evaluator = BinaryClassificationEvaluator(metricName=eval_type)
            else:
                evaluator = MulticlassClassificationEvaluator(metricName=eval_type)

            rf_cv = CrossValidator(
                estimator=rf,
                evaluator=evaluator,
                estimatorParamMaps=param_grid
            )

            fitted_model = rf_cv.fit(train_data)
            return fitted_model
            
        except Exception as e:
            logger.error(f"Error training Random Forest Model: {str(e)}", stacklevel=2)
            raise

    def save_model(self, model, models_path: str):
        """Save the trained model."""
        try:
            os.makedirs(os.path.dirname(models_path), exist_ok=True)
            model_name = self.config.get("model_name", "trained_model")
            model_path = os.path.join(models_path, model_name)
            model.save(model_path)
            logger.info(f"Model saved successfully to {model_path}")

        except Exception as e:
            logger.error(f"Error saving trained model: {str(e)}", stacklevel=2)
            raise


def main():
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
    main()
