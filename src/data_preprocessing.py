from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *

import logging
import yaml
import os

class Log:

    @staticmethod
    def setup_logging(log_dir="logging-info"):
        os.makedirs(log_dir, exist_ok=True)

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        if not logger.hasHandlers():
            file_handler = logging.FileHandler(
                filename=os.path.join(log_dir, "logs.log"), 
                encoding="utf-8"
            )
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

logger = Log.setup_logging()


class LoadYamlParams:
    """Load parameters from YAML file."""

    def load_params(self, filepath: str) -> dict:

        try:
            with open(filepath, "r") as file:
                params = yaml.safe_load(file)
            logger.info(f"Parameters loaded successfully from {filepath}", stacklevel=2)
            return params
        
        except Exception as e:
            logger.error(f"Error loading parameters: {str(e)}", stacklevel=2)
            raise


class SparkLoader:
    """
    This class creates or loads the Spark session.

    Args:
        app_name (str): Name of application

    Attributes:
        spark: SparkSession
    """

    def __init__(self, app_name):
        logger.info(
            msg=f"Initializing SparkLoader Class with app name: {app_name}",
            stacklevel=2        
        )
        try:
            logger.debug("Creating Spark session...", stacklevel=2)

            self.spark = SparkSession.builder \
                .appName(app_name) \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.parquet.enableVectorizedRender", "true") \
                .config("spark.sql.parquet.columnarReaderBatchSize", "4096") \
                .getOrCreate()
            
            logger.info("Spark session created successfully", stacklevel=2)

        except Exception as e:
            logger.error(f"Error creating Spark session: {str(e)}", stacklevel=2)
            raise e
        
    
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
    
    def save_data_parquet(self, df: DataFrame, output_path: str):
        """
        Save processed data as Parquet files for downstream use.

        Args:
            df (Dataframe): Input DataFrame
            output_path (str): Relative file path to save the dataframe to
        """

        logger.info(f"[INFO] Saving {df.count()} records as Parquet files to {output_path}...")

        try:
            os.makedirs(output_path, exist_ok=True)

            partitions = self.config.get("partitions", [])
            if partitions:
                df.write.parquet(output_path, mode="overwrite", partitionBy=partitions)
            else:
                df.write.parquet(output_path, mode="overwrite")

            logger.info(f"[INFO] Successfully wrote {df.count()} records")
            logger.info(f"[INFO] Output location: {output_path}")

            return df.count()

        except Exception as e:
            logger.error(f"[ERROR] Error saving data as Parquet files: {str(e)}", stacklevel=2)
            raise



def process_data_():
    """Main function to run the preprocessing pipeline."""

    logger.info("[INFO] Logging setup completed successfully")

    try:
        params_obj = LoadYamlParams()
        params = params_obj.load_params(filepath="params.yaml")

        app_name = params["sparksession"]["name"]
        raw_data_file = os.path.join(
            params["paths"]["raw_data_path"],
            params["paths"]["raw_data_file"]
        )
        processed_data_path = params["paths"]["processed_data_path"]

        spark_loader = SparkLoader(app_name)
        preprocessor = DataPreprocessor(
            spark=spark_loader.spark,
            config=params["data_preprocessing"]
        )
        df = preprocessor.load_data_csv(
            input_path=raw_data_file
        )
        preprocessor.check_null_data(df=df)
        df_cleaned = preprocessor.clean_data(df=df)
        preprocessor.save_data_parquet(
            df=df_cleaned, 
            output_path=processed_data_path,
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
