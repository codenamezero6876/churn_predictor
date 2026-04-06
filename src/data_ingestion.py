from pyspark.sql import SparkSession, DataFrame
from src.helper_class import Log, LoadYamlParams, SparkLoader

logger = Log.setup_logging()


class DataIngestor:
    """This class handles data ingestion."""

    def __init__(self, spark: SparkSession, config: dict):
        """
        Initialize data ingestor with SparkSession.

        Args:
            spark (SparkSession): Existing SparkSession to use
            config (dict): Data ingestor configurations to use
        """
        self.spark = spark
        self.config = config
        logger.info(
            "[INFO] Data ingestor initialized with existing Spark session"
        )

    def ingest_source_data_from_adls(
        self,
        source_path: str,
        schema_path: str,
        file_name: str = "customers"
    ):
        """Reads streaming data from ADLS according to specified format."""
        try:
            logger.info(["[INFO] Ingesting source data from ADLS..."])
            format = self.config.get("save_format", "parquet")
            df = self.spark.readStream.format("cloudFiles") \
                .option("cloudFiles.format", format) \
                .option("cloudFiles.schemaLocation", f"{schema_path}/cp_{file_name}") \
                .load(f"{source_path}/{file_name}")
            logger.info("[INFO] Source data ingested successfuly")
            return df

        except Exception as e:
            logger.error(f"[ERROR] Error ingesting source data from ADLS: {str(e)}", stacklevel=2)
            raise

    def ingest_incremental_data(
        self,
        df: DataFrame,
        source_path: str,
        file_name: str = "customers"
    ):
        """Writes and saves incremental data according to specified format."""
        try:
            logger.info("[INFO] Ingesting incremental data...")
            format = self.config.get("save_format", "parquet")
            df.writeStream.format(format) \
                .outputMode("append") \
                .option("checkpointLocation", f"{source_path}/cp_{file_name}") \
                .option("path", f"{source_path}/{file_name}") \
                .trigger(once=True) \
                .start()
            logger.info("[INFO] Incremental data ingested successfully")

        except Exception as e:
            logger.error(f"[ERROR] Error ingesting incremental data: {str(e)}", stacklevel=2)
            raise



def ingest_data_():
    """Main function to run data ingestion pipeline."""

    logger.info("[INFO] Initializing Data Ingestion Pipeline...")

    try:
        params_obj = LoadYamlParams()
        params = params_obj.load_params(filepath="params.yaml")

        app_name = params["sparksession"]["name"]
        raw_data_path = params["adls_paths"]["data"]["raw"]
        ingested_data_path = params["adls_paths"]["data"]["ingested"]

        spark_loader = SparkLoader(app_name)

        ingestor = DataIngestor(
            spark=spark_loader.spark,
            config=params["data_ingestion"]
        )
        df = ingestor.ingest_source_data_from_adls(
            source_path=raw_data_path,
            schema_path=ingested_data_path
        )
        ingestor.ingest_incremental_data(
            df=df,
            source_path=ingested_data_path
        )
        logger.info("[INFO] Data Ingestion Pipeline completed successfully")

    except Exception as e:
        logger.error(f"[ERROR] Data Ingestion Pipeline failed: {str(e)}", stacklevel=2)
        raise

    finally:
        spark_loader.spark.stop()
        logger.info("Spark session closed")


if __name__ == "__main__":
    ingest_data_()
