
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator



# Script to train the ALS model

def get_raw_df(spark_session, file_name):
  df_raw  = spark_session.read.csv(file_name, header=True, inferSchema=True)
  return df_raw


def get_counts(df):
  # Preliminary Statistics
  ratings_counts = df.select("rating").count()
  users_count = df.select("userId").distinct().count()
  movies_count = df.select("movieId").distinct().count()
  ratings_by_userId = df.groupBy("userId").count().show()

  print(f"Total Ratings: {ratings_counts}\n Total user: {users_count}\n Movies count: {movies_count} \nRatings by User: {ratings_by_userId}")


def train_ALS_model(df, userColumn, itemColumn, ratingColumn, max_iterations, latent_factors):
  
  # Divide in training/test datasets
  (train, test) = df.randomSplit([0.8, 0.2], seed=42)
  
  # Define ALS model
  als = ALS(
    maxIter=max_iterations,
    regParam=0.1,
    rank=latent_factors,
    userCol=userColumn,
    itemCol=itemColumn,
    ratingCol=ratingColumn,
    coldStartStrategy="drop", 
    implicitPrefs=False, 
    )
  
  # Train model
  print("Training ALS model ....\n")
  model = als.fit(train)

  # Test model
  print("Getting predictions for evaluation....\n")
  predictions = model.transform(test)

  evaluator = RegressionEvaluator(
      metricName="rmse",
      labelCol="rating",
      predictionCol="prediction"
  )

  print("Getting Evaluation Parameters")
  rmse = evaluator.evaluate(predictions)
  print(f"Root-mean-square error = {rmse:.4f}")

  return model
  


if __name__ == "__main__":

  spark = SparkSession.builder.appName("ALS Model Creation").getOrCreate()

  df_ratings_small = get_raw_df(spark, "data/movie-lens-small-latest-dataset/ratings.csv")

  als_model_25 = train_ALS_model(df_ratings_small, "userId", "movieId", "rating", 10, 25)

  # Saving model for later use
  print("\nSaving model...")
  als_model_25.write().overwrite().save("models/ratings_small_model-latent-features-25")
  print("\n\nModel saved")