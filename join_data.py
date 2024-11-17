from pyspark.sql import SparkSession
from pyspark.sql import functions as F

"""
This code is part of the pre-processing process which destructures the concatenated sarcastic labels
and join the sarcastic with the comment information with the corresponding ID.
"""

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Join Table") \
    .config("spark.driver.memory", "32g") \
    .config("spark.executor.memory", "5g") \
    .config("spark.executor.cores", "4") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.default.parallelism", "200") \
    .getOrCreate()

# Load datasets
df = spark.read.format('csv').options(header="true", inferSchema="true") \
               .load("s3://bigdataprojectreddit/athena_results/Unsaved/2024/10/29/0401beaa-a78b-4852-a168-1c7e60650414.csv")
df.show()

df2 = spark.read.format('csv').options(header="false", inferSchema="true") \
                .load("s3://bigdataprojectreddit/data/sarc_database/sarc_csv/break_label1.csv/part-00000-1993baf7-c382-440b-ad8f-5ea77d579686-c000.csv")
df2 = df2.orderBy(F.rand()).limit(1000000)
# Apply column names
column_names = ["post&comment_id", "response_id", "label"]
df2 = df2.toDF(*column_names)
df2.show()
print(df2.count())

# Filter rows as needed
df2.filter(F.col("response_id").like("%d16l869%")).show()
df2.filter(F.col("post&comment_id").like("%d16l869%")).show()

# Perform joins
df_join_1 = df.join(df2, df2["post&comment_id"].contains(df["id"]), "left") \
              .select(df["*"], df2["label"])

df_join_2 = df.join(df2, df2["response_id"].contains(df["id"]), "left") \
              .select(df["*"], df2["label"])

# Combine both joins with union and drop duplicates
df_combined = df_join_1.union(df_join_2).dropDuplicates()

# Show combined data
df_combined.show()

# Write combined data to S3
df_combined.write.mode("overwrite").option("header", "true") \
            .csv("s3://bigdataprojectreddit/data/processed_data/")
