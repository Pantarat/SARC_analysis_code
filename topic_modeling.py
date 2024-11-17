from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, CountVectorizer
from pyspark.ml.clustering import LDA
from pyspark.sql.functions import col, udf, array_max, array_position, count, desc
from pyspark.sql.types import IntegerType
import pyspark.sql.functions as F

"""
This code is for modeling the clusters of comments using Linear Discriminant Analysis (LDA)
to predict the topic and comparing the resulting cluster with the subreddit that the comment
is contained in.
"""

# Initialize Spark session
spark = SparkSession.builder \
    .appName("TopicModeling") \
    .config("spark.driver.memory", "64g") \
    .config("spark.driver.maxResultSize", "32g") \
    .config("spark.executor.memory", "16g") \
    .config("spark.executor.cores", "4") \
    .config("spark.sql.shuffle.partitions", "400") \
    .config("spark.default.parallelism", "400") \
    .getOrCreate()

# Load data
data = spark.read.format('csv').options(header="true", inferSchema="true") \
               .load("s3://bigdataprojectreddit/athena_results/Unsaved/2024/10/29/0401beaa-a78b-4852-a168-1c7e60650414.csv")

# Step 1: Tokenization
tokenizer = Tokenizer(inputCol="text", outputCol="words")
words_data = tokenizer.transform(data)

# Step 2: Vectorization
cv = CountVectorizer(inputCol="words", outputCol="features", vocabSize=100000, minDF=2)
cv_model = cv.fit(words_data)
vectorized_data = cv_model.transform(words_data)

# Step 3: Train the LDA model
lda = LDA(k=100, maxIter=20, featuresCol="features", seed=1)
lda_model = lda.fit(vectorized_data)

# Step 4: Transform data to get topic distribution
transformed_data = lda_model.transform(vectorized_data)

# Step 5: Extract the predicted topic
@udf(returnType=IntegerType())
def get_max_topic_index(topic_dist):
    return int(max(enumerate(topic_dist), key=lambda x: x[1])[0])

# Apply the UDF to get the predicted topic
transformed_data = transformed_data.withColumn(
    "predicted_topic",
    get_max_topic_index(F.col("topicDistribution"))
)

# Step 6: Join with original data
output_data = data.join(
    transformed_data.select("id", "predicted_topic"), 
    on="id", 
    how="left"
)

# Step 7: Calculate and display topic distribution
topic_distribution = transformed_data.groupBy("predicted_topic") \
    .agg(count("*").alias("text_count")) \
    .orderBy(desc("text_count"))

# Display topic distribution
print("Topic Distribution Summary:")
topic_distribution.show(100)

# Optional: Save topic distribution to a separate file
topic_distribution.write.mode("overwrite").option("header", "true") \
    .csv("s3://bigdataprojectreddit/data/topic_distribution/")

# Step 8: Save the main output
output_data.write.mode("overwrite").option("header", "true") \
           .csv("s3://bigdataprojectreddit/data/predicted_topics/")

output_data.show()

# Step 9: Calculate and display some basic statistics
total_texts = topic_distribution.select(F.sum("text_count")).collect()[0][0]
avg_texts_per_topic = topic_distribution.select(F.avg("text_count")).collect()[0][0]
max_texts_in_topic = topic_distribution.select(F.max("text_count")).collect()[0][0]
min_texts_in_topic = topic_distribution.select(F.min("text_count")).collect()[0][0]

print("\nTopic Distribution Statistics:")
print(f"Total number of comments: {total_texts}")
print(f"Average comments per topic: {avg_texts_per_topic:.2f}")
print(f"Maximum comments in a topic: {max_texts_in_topic}")
print(f"Minimum comments in a topic: {min_texts_in_topic}")

# Stop Spark session
# spark.stop()