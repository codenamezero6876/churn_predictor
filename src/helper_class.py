from pyspark.sql import SparkSession, DataFrame
import os, yaml, logging, typing


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


def load_data(spark: SparkSession, input_path: str, format: str="parquet") -> DataFrame:
    try:
        df = spark.read.format(format).load(input_path)
        logger.info(f"[INFO] Data successfully loaded from {input_path}")
        return df

    except Exception as e:
        logger.error(f"[ERROR] Error loading files: {str(e)}", stacklevel=2)
        raise

def save_data(
        df: DataFrame, 
        output_path: str, 
        format: str="parquet",
        partitions: typing.List[str]=[],
        local: bool=False
    ):
    """
    Save processed data for downstream use.

    Args:
        df (Dataframe): Input DataFrame
        output_path (str): Relative file path to save the dataframe to
    """

    logger.info(f"[INFO] Saving {df.count()} records to {output_path}...")

    try:
        if local:
            os.makedirs(output_path, exist_ok=True)
        
        if partitions:
            df.write \
                .format(format) \
                .mode("overwrite") \
                .option("path", output_path) \
                .partitionBy(*partitions) \
                .save()
        else:
            df.write \
                .format(format) \
                .mode("overwrite") \
                .option("path", output_path) \
                .save()

        logger.info(f"[INFO] Successfully wrote {df.count()} records")
        logger.info(f"[INFO] Output location: {output_path}")

        return df.count()

    except Exception as e:
        logger.error(f"[ERROR] Error saving data: {str(e)}", stacklevel=2)
        raise
