from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession, DataFrame, Column, functions as F

from typing import Tuple
from functools import reduce
from src.data_preprocessing import Log, SparkLoader, LoadYamlParams


logger = Log.setup_logging()


class FeatureEngineer:
    """
    Handles feature engineering operations for the churn prediction model.
    """

    def __init__(self, spark: SparkSession, config: dict):
        """
        Initialize feature engineer with SparkSession.

        Args:
            spark (SparkSession): Existing SparkSession to use
            config (dict): Feature engineering configurations to use
        """
        self.spark = spark
        self.config = config
        logger.info("[INFO] FeatureEngineer initialized with existing SparkSession", stacklevel=2)

    def load_data_parquet(self, input_path: str) -> DataFrame:
        """Load processed data from local Parquet files."""

        try:
            df = self.spark.read.parquet(input_path)
            logger.info(f"[INFO] Processed data successfully loaded from {input_path}")
            return df

        except Exception as e:
            logger.error(f"[ERROR] Error loading files: {str(e)}", stacklevel=2)
            raise

    def categorize_existing_columns(self, df: DataFrame) -> DataFrame:
        """
        Convert existing numeric columns specified in the config into a categorical one.

        Args:
            df (DataFrame): Input DataFrame
        """

        def make_condition(category: str, c: Column, label: dict):
            if label["start"] == "null":
                return c.when(F.col(category) < label["end"], label["name"])
            elif label["end"] == "null":
                return c.when(F.col(category) > label["start"], label["name"])
            else:
                return c.when(F.col(category).between(label["start"], label["end"]), label["name"])

        try:
            if (
                self.config.get("new_features") and
                self.config.get("new_features").get("categorize_from_existing")
            ):
                features = self.config.get("new_features").get("categorize_from_existing")

                for feature in features:
                    category = feature["col_name"]
                    labels = feature["labels"]
                    cond = reduce(
                        lambda c, label: make_condition(category, c, label),
                        labels,
                        F
                    )

                    df = df.withColumn(f"{category}_category", cond)

                logger.info("[INFO] Successfully categorized all existing numeric columns specified in config")
                return df

        except Exception as e:
            logger.error(f"[ERROR] Error categorizing numeric column: {str(e)}", stacklevel=2)
            raise

    def create_some_feature(self, df: DataFrame) -> DataFrame:
        """
        Create some new feature using a formula or some shit.
        """

        try:
            return df
        
        except Exception as e:
            logger.error(f"[ERROR] Error creating this feature: {str(e)}", stacklevel=2)
            raise

    def create_ml_features(self, df: DataFrame) -> DataFrame:
        """
        Create machine learning ready features.
        """

        try:
            categorical_columns = [
                col for col in df.columns
                if df.schema[col].dataType.typeName() == 'string'
            ]

            numerical_columns = [
                col for col in df.columns
                if col not in categorical_columns + ["Churn"]
            ]

            stages = []

            for col in categorical_columns:
                indexer = StringIndexer(
                    inputCol=col,
                    outputCol=f"{col}_idx",
                    handleInvalid="keep"
                )
                encoder = OneHotEncoder(
                    inputCols=[f"{col}_idx"],
                    outputCols=[f"{col}_vec"]
                )
                stages.extend([indexer, encoder])

            assembler_input = [f"{col}_vec" for col in categorical_columns] + numerical_columns
            assembler = VectorAssembler(
                inputCols=assembler_input,
                outputCol="features_raw",
                handleInvalid="keep"
            )
            stages.append(assembler)

            scaler = StandardScaler(
                inputCol="features_raw",
                outputCol="features",
                withStd=True,
                withMean=True
            )
            stages.append(scaler)

            pipeline = Pipeline(stages=stages)
            model = pipeline.fit(df)
            processed_df = model.transform(df)

            df_result = processed_df.select(
                "features",
                F.col("Churn").cast("double").alias("label")
            )

            logger.info("[INFO] Successfully created ML features")
            return df_result

        except Exception as e:
            logger.error(f"[ERROR] Error creating ML-ready features: {str(e)}", stacklevel=2)
            raise

    def split_data(self, df: DataFrame) -> Tuple[DataFrame]:
        """
        Splits the dataframe into train and test dataframes.
        Train-test split ratio defaults to 0.2 unless specified in config.
        Random seed defaults to 42 unless specified in config.
        """
        try:
            split_ratio = self.config.get("train_test_split_ratio", 0.2)
            random_seed = self.config.get("random_seed", 42)
            train_df, test_df = df.randomSplit(
                weights=[1 - split_ratio, split_ratio],
                seed=random_seed
            )
            logger.info("[INFO] Train-test split is successful")
            return train_df, test_df
        
        except Exception as e:
            logger.error(f"[ERROR] Error splitting data: {str(e)}", stacklevel=2)
            raise

    def save_feature_data_parquet(
        self,
        train_df: DataFrame,
        test_df: DataFrame,
        train_path: str,
        test_path: str
    ) -> None:
        """
        Save train and test DataFrames as Parquet files
        """
        try:
            train_df.coalesce(1) \
                .write \
                .mode("overwrite") \
                .format("parquet") \
                .save(train_path)
            
            test_df.coalesce(1) \
                .write \
                .mode("overwrite") \
                .format("parquet") \
                .save(test_path)
            
            logger.info(f"[INFO] Train data saved to {train_path}.")
            logger.info(f"[INFO] Test data saved to {test_path}.")

        except Exception as e:
            logger.error(f"[ERROR] Error saving feature data: {str(e)}", stacklevel=2)
            raise


def engineer_features_():
    """Main function to run the feature engineering pipeline."""

    logger.info("[INFO] Starting feature engineering pipeline...")

    try:
        params_obj = LoadYamlParams()
        params = params_obj.load_params(filepath="params.yaml")

        app_name = params["sparksession"]["name"]
        processed_data_path = params["paths"]["processed_data_path"]
        train_path = params["paths"]["train_data_path"]
        test_path = params["paths"]["test_data_path"]

        spark_loader = SparkLoader(app_name)
        engineer = FeatureEngineer(
            spark=spark_loader.spark,
            config=params["feature_engineering"]
        )
        df = engineer.load_data_parquet(
            input_path=processed_data_path
        )
        df = engineer.categorize_existing_columns(df)
        df = engineer.create_ml_features(df)
        train_df, test_df = engineer.split_data(df)
        engineer.save_feature_data_parquet(
            train_df=train_df,
            test_df=test_df,
            train_path=train_path,
            test_path=test_path
        )

        logger.info(f"[INFO] Feature engineering pipeline completed successfully")

    except Exception as e:
        logger.error(f"[ERROR] Feature engineering pipeline failed: {str(e)}", stacklevel=2)
        raise

    finally:
        spark_loader.spark.stop()
        logger.info("Spark session closed")


if __name__ == "__main__":
    engineer_features_()
