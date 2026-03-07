import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import from_json, col, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType

# 1. Initialize Glue Context
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# 2. Configuration - Hardcoded for your single environment setup
# In a multi-stack setup, these are passed as job parameters
KINESIS_STREAM_NAME = "sentinel-fraud-detection-transaction-stream"
S3_BUCKET_PATH = "s3://sentinel-fraud-detection-datalake" # Update with your actual bucket name
CHECKPOINT_LOCATION = f"{S3_BUCKET_PATH}/temp/checkpoint/"

# 3. Define the Schema for incoming JSON transactions
transaction_schema = StructType([
    StructField("transaction_id", StringType(), True),
    StructField("card_number", StringType(), True),
    StructField("terminal_id", StringType(), True),
    StructField("transaction_amount", DoubleType(), True),
    StructField("transaction_timestamp", StringType(), True),
    StructField("city", StringType(), True)
])

# 4. Read Stream from Kinesis
raw_stream_df = spark.readStream \
    .format("kinesis") \
    .option("streamName", KINESIS_STREAM_NAME) \
    .option("endpointUrl", "https://kinesis.us-east-1.amazonaws.com") \
    .option("startingPosition", "latest") \
    .load()

# 5. Transformation Logic
# Bronze Layer: Cast data to String (Raw format)
bronze_df = raw_stream_df.selectExpr("CAST(data AS STRING) as raw_data", "approximateArrivalTimestamp as arrival_time")

# Silver Layer: Parse JSON and Flatten
silver_df = bronze_df.select(
    from_json(col("raw_data"), transaction_schema).alias("data"),
    col("arrival_time")
).select("data.*", "arrival_time") \
 .withColumn("ingestion_timestamp", current_timestamp())

# 6. Writing to S3 (Medallion Folders)
# --- Bronze Layer: Write Raw JSON strings ---
bronze_query = bronze_df.writeStream \
    .format("parquet") \
    .option("path", f"{S3_BUCKET_PATH}/bronze/") \
    .option("checkpointLocation", f"{CHECKPOINT_LOCATION}bronze/") \
    .outputMode("append") \
    .start()

# --- Silver Layer: Write Cleaned Parquet data ---
query = silver_df.writeStream \
    .format("parquet") \
    .option("path", f"{S3_BUCKET_PATH}/silver/") \
    .option("checkpointLocation", CHECKPOINT_LOCATION) \
    .outputMode("append") \
    .start()

# Wait for both streams to terminate
spark.streams.awaitAnyTermination()