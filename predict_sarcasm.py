from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col
from datetime import datetime
import logging
import sys

"""
This code is designed to train a sarcastic detection model from the given text data.
"""

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

try:
    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("SarcasmDetection") \
        .config("spark.driver.memory", "64g") \
        .config("spark.executor.memory", "16g") \
        .config("spark.executor.cores", "16") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.default.parallelism", "200") \
        .getOrCreate()
    
    logger.info("Spark session created successfully")

    # S3 paths
    input_path = "s3://bigdataprojectreddit/data/comment-id-with-sarcastic-label/774ce39a-f9cf-4f7b-8909-cb730ece185c.csv"
    s3_base_path = "s3://bigdataprojectreddit/ml-output-data/Sarcasm_Classification/"

    # Load data
    logger.info(f"Loading data from: {input_path}")
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    
    # Filter the DataFrame for rows where label is 0 or 1
    df = df.filter((col("label") == 1) | (col("label") == 0))

    # Verify the filtered data
    df.groupBy("label").count().show()
    
    df.cache()

    # Basic data validation
    if df.count() == 0:
        raise ValueError("Empty dataset")
    
    if not set(['text', 'label']).issubset(df.columns):
        raise ValueError("Missing required columns: text and label")

    # Print data summary
    logger.info("\nDataset Summary:")
    total_records = df.count()
    logger.info(f"Total Records: {total_records}")
    
    logger.info("\nLabel Distribution:")
    df.groupBy("label").count().show()
    
    logger.info("\nSample Records:")
    df.show(5, truncate=False)
    
    logger.info("\nText Length Statistics:")
    df.selectExpr("length(text) as text_length").describe().show()

    # Split the data
    train_df, test_df = df.randomSplit([0.9, 0.1], seed=42)
    train_df.cache()
    test_df.cache()
    
    logger.info(f"\nTraining Set Size: {train_df.count()}")
    logger.info(f"Test Set Size: {test_df.count()}")

    # Create pipeline
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
    idf = IDF(inputCol="raw_features", outputCol="features")
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20)
    
    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])

    # Create param grid for cross validation
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.01, 0.1, 0.3]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
        .build()

    # Create cross validator
    crossval = CrossValidator(estimator=pipeline,
                            estimatorParamMaps=paramGrid,
                            evaluator=BinaryClassificationEvaluator(),
                            numFolds=3,
                            parallelism=4)

    # Train the model and track time
    logger.info("Starting model training...")
    start_time = datetime.now()
    cv_model = crossval.fit(train_df)
    training_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Model training completed in {training_time:.2f} seconds")

    # Make predictions on test data
    predictions = cv_model.transform(test_df)
    predictions.cache()

    # Calculate metrics
    binary_evaluator = BinaryClassificationEvaluator()
    multi_evaluator = MulticlassClassificationEvaluator()

    # Get the best model's metrics
    best_model = cv_model.bestModel
    lr_model = best_model.stages[-1]  # Get the LogisticRegression model from the pipeline

    metrics = {
        "auc": float(binary_evaluator.evaluate(predictions)),
        "accuracy": float(multi_evaluator.setMetricName("accuracy").evaluate(predictions)),
        "precision": float(multi_evaluator.setMetricName("weightedPrecision").evaluate(predictions)),
        "recall": float(multi_evaluator.setMetricName("weightedRecall").evaluate(predictions)),
        "f1": float(multi_evaluator.setMetricName("f1").evaluate(predictions)),
        "training_time": training_time,
        "timestamp": datetime.now().isoformat(),
        "training_losses": lr_model.summary.objectiveHistory,  # Get training losses from the best model
        "num_training_examples": train_df.count(),
        "num_test_examples": test_df.count()
    }

    # Generate timestamp for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model and metrics to S3
    model_path = f"{s3_base_path}/models/{timestamp}"
    metrics_path = f"{s3_base_path}/metrics/{timestamp}"
    
    logger.info(f"Saving model to: {model_path}")
    cv_model.save(model_path)
    
    logger.info(f"Saving metrics to: {metrics_path}")
    metrics_df = spark.createDataFrame([metrics])
    metrics_df.write.mode("overwrite").json(metrics_path)

    # Print results
    logger.info("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"{metric}: {value:.4f}")

    logger.info("\nConfusion Matrix:")
    predictions.groupBy("label", "prediction").count().show()

    logger.info("\nSample Predictions:")
    predictions.select("text", "label", "prediction", "probability").show(5, truncate=False)

    logger.info("Pipeline completed successfully")

except Exception as e:
    logger.error(f"Pipeline failed: {str(e)}")
    raise

finally:
    # Clean up
    spark.stop()
    logger.info("Spark session stopped")