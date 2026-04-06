from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *

import os
from src.helper_class import \
    Log, \
    LoadYamlParams, \
    SparkLoader, \
    load_data, \
    save_data

logger = Log.setup_logging()
    
class DataPreprocessor:
    """
    This class handles data preprocessing and cleaning operations.
    """

    def __init__(self, spark: SparkSession, config: dict):
        """
        Initialize preprocessor with SparkSession.

        Args:
            spark (SparkSession): Existing SparkSession to use
            config (dict): Preprocessing configurations to use
        """
        self.spark = spark
        self.config = config
        logger.info(
            "[INFO] Data preprocessor initialized with existing Spark session"
        )

    def load_data_csv(self, input_path: str, show_schema=True) -> DataFrame:
        """Load data from local CSV file."""

        try:
            df = self.spark.read.csv(
                path=input_path,
                header=True,
                inferSchema=True,
                mode="PERMISSIVE"
            )
            logger.info(f"[INFO] Data loaded successfully from {input_path}")
            logger.info(f"[INFO] Total data count: {df.count()}")


            if show_schema:
                for field in df.schema:
                    logger.info(f"[INFO] Column Name: {field.name:<20} | DataType: {str(field.dataType):<20} | Nullable: {field.nullable}")

            return df
        
        except Exception as e:
            logger.error(f"[ERROR] Error loading data: {str(e)}", stacklevel=2)
            raise

    def load_ingested_data(self, input_path: str) -> DataFrame:
        load_format = self.config.get("load_format", "parquet")
        df = load_data(
            spark=self.spark,
            input_path=input_path,
            format=load_format
        )
        return df
        
    def check_null_data(self, df: DataFrame) -> None:
        """Analyze and log null values in the datset."""

        try:
            total_count = float(df.count())
            column_types_dict = dict(df.dtypes)

            for column in df.columns:
                col_type = column_types_dict[column]

                if col_type in ("int", "double", "float", "long"):
                    null_condition = col(column).isNull()
                    
                    if col_type != "int":
                        null_condition = null_condition | isnan(col(column))

                else:
                    null_condition = (
                        col(column).isNull() |
                        (col(column) == "") |
                        (col(column) == "NULL") |
                        (col(column) == "null")
                    )

                null_count = df.filter(null_condition).count()
                percentage = (null_count / total_count) * 100

                logger.info(f"[INFO] Column: {column:<40} | Null count: {null_count:<15} | Percentage: {percentage:.2f}%")

        except Exception as e:
            logger.error(f"[ERROR] Error analyzing null values: {str(e)}", stacklevel=2)
            raise

    def clean_data(self, df: DataFrame) -> DataFrame:
        """
        Clean the dataset based on specified parameters.

        Args:
            df (DataFrame): Input DataFrame
        """

        try:
            columns_to_drop = self.config.get("columns_to_drop", [])
            if columns_to_drop:
                df = df.drop(*columns_to_drop)
                logger.info(f"[INFO] Dropped columns: {columns_to_drop}")
            else:
                logger.info("[INFO] No columns were dropped")

            columns_to_dropna = self.config.get("columns_to_dropna", [])
            if columns_to_dropna:
                df = df.dropna(subset=[*columns_to_dropna])
                logger.info(f"[INFO] Dropped rows with missing values in columns: {columns_to_dropna}")
            else:
                logger.info(f"[INFO] No rows with missing values in specified columns")
            logger.info(f"[INFO] Data count after handling missing values: {df.count()}")

            columns_to_dedup = self.config.get("columns_to_dedup", [])
            if "columns_to_dedup" in self.config:
                df = df.dropDuplicates([*columns_to_dedup])
                logger.info(f"[INFO] Removed duplicate values from columns: {columns_to_dedup}")
            else:
                df = df.dropDuplicates()
            logger.info(f"[INFO] Data count after removing duplicates: {df.count()}")

            return df
        
        except Exception as e:
            logger.error(f"[ERROR] Error cleaning data: {str(e)}", stacklevel=2)
            raise

    def save_preprocessed_data(self, df: DataFrame, output_path: str):
        save_format = self.config.get("save_format", "parquet")
        partitions = self.config.get("partitions", [])
        is_local = self.config.get("is_local", False)

        save_data(
            df=df,
            output_path=output_path,
            format=save_format,
            partitions=partitions,
            local=is_local
        )

def process_data_():
    """Main function to run the preprocessing pipeline."""

    logger.info("[INFO] Logging setup completed successfully")

    try:
        params_obj = LoadYamlParams()
        params = params_obj.load_params(filepath="params.yaml")

        app_name = params["sparksession"]["name"]

        ingested_data_path = params["paths"]["data"]["ingested"]
        processed_data_path = params["paths"]["data"]["processed"]

        spark_loader = SparkLoader(app_name)
        preprocessor = DataPreprocessor(
            spark=spark_loader.spark,
            config=params["data_preprocessing"]
        )
        df = preprocessor.load_ingested_data(input_path=ingested_data_path)
        preprocessor.check_null_data(df=df)
        df_cleaned = preprocessor.clean_data(df=df)
        preprocessor.save_preprocessed_data(
            df=df_cleaned,
            output_path=processed_data_path
        )
        logger.info("[INFO] Preprocessing pipeline completed successfully")

    except Exception as e:
        logger.error(f"[ERROR] Preprocessing pipeline failed: {str(e)}", stacklevel=2)
        raise

    finally:
        spark_loader.spark.stop()
        logger.info("Spark session closed")


if __name__ == "__main__":
    process_data_()
