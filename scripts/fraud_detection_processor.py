import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import from_json, col, current_timestamp, date_format, window, sum as _sum, count as _count, when as _when, first as _first
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
# Coalesce to reduce partition fragmentation and add date partitioning for crawler consolidation
bronze_with_partition = bronze_df.withColumn("partition_date", date_format(col("arrival_time"), "yyyy-MM-dd"))

bronze_query = bronze_with_partition.coalesce(10).writeStream \
    .partitionBy("partition_date") \
    .format("parquet") \
    .option("path", f"{S3_BUCKET_PATH}/bronze/") \
    .option("checkpointLocation", f"{CHECKPOINT_LOCATION}bronze/") \
    .outputMode("append") \
    .start()

# --- Silver Layer: Write Cleaned Parquet data ---
# Partition by date for efficient querying while maintaining single table in Glue Catalog
silver_with_partition = silver_df.withColumn("partition_date", date_format(col("ingestion_timestamp"), "yyyy-MM-dd"))

query = silver_with_partition.coalesce(10).writeStream \
    .partitionBy("partition_date") \
    .format("parquet") \
    .option("path", f"{S3_BUCKET_PATH}/silver/") \
    .option("checkpointLocation", CHECKPOINT_LOCATION) \
    .outputMode("append") \
    .start()

# --- Gold Layer: Real-Time Fraud Aggregations ---
# Windowed aggregations to detect velocity-based fraud patterns
# Watermarking: Waits 10 minutes for late-arriving data before finalizing results
gold_df = silver_df \
    .withWatermark("ingestion_timestamp", "10 minutes") \
    .groupBy(
        window(col("ingestion_timestamp"), "10 minutes", "5 minutes"),
        col("card_number"),
        col("city")
    ).agg(
        _sum("transaction_amount").alias("total_spent"),
        _count("transaction_id").alias("transaction_count"),
        _first("terminal_id").alias("terminal_id")
    ) \
    .withColumn("avg_transaction_value", col("total_spent") / col("transaction_count")) \
    .withColumn("fraud_risk_flag", 
        _when((col("total_spent") > 10000) | (col("transaction_count") > 10), "HIGH")
        .when((col("total_spent") > 5000) | (col("transaction_count") > 5), "MEDIUM")
        .otherwise("LOW")
    ) \
    .withColumn("partition_date", date_format(col("window").getField("start"), "yyyy-MM-dd"))

gold_query = gold_df.coalesce(5).writeStream \
    .partitionBy("partition_date") \
    .format("parquet") \
    .option("path", f"{S3_BUCKET_PATH}/gold/") \
    .option("checkpointLocation", f"{CHECKPOINT_LOCATION}gold/") \
    .outputMode("complete") \
    .start()

# Wait for both streams to terminate
spark.streams.awaitAnyTermination()