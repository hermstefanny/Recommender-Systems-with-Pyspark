
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
import pyspark.sql.functions as sqlf
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import Row
import numpy as np


# Script to train the ALS model

def get_raw_df(spark_session, file_name):
  df_raw  = spark_session.read.csv(file_name, header=True, inferSchema=True)
  return df_raw


def get_counts(df):
  '''Preliminary Statistics'''
  ratings_counts = df.select("rating").count()
  users_count = df.select("userId").distinct().count()
  movies_count = df.select("movieId").distinct().count()
  ratings_by_userId = df.groupBy("userId").count().show()

  print(f"Total Ratings: {ratings_counts}\n Total user: {users_count}\n Movies count: {movies_count} \nRatings by User: {ratings_by_userId}")


def train_ALS_model(df, userColumn, itemColumn, ratingColumn, max_iterations=20, latent_factors=45):
  
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
  


def get_new_prediction(new_ratings_df, model, new_user_id=3000, top_n=10):
    '''
    Cold-start ALS: recommend for a new user without retraining.
    '''
    # Join new ratings with item factors
    rated = new_ratings_df.join(model.itemFactors,
                                new_ratings_df.movieId == model.itemFactors.id,
                                "inner").select("rating", "features")

    # Build synthetic user vector (weighted avg)
    rows = rated.collect()
    num = sum(np.array(r.features) * r.rating for r in rows)
    den = sum(r.rating for r in rows)
    try:
      user_vector = num / den
    except Exception as e:
      print (e)
       

    # UDF for dot product
    def dot(features):
        return float(np.dot(user_vector, features))
    dot_udf = sqlf.udf(dot, "double")

    # Predict scores for all items
    preds = model.itemFactors.withColumn("prediction", dot_udf("features")) \
                             .withColumn("userId", sqlf.lit(new_user_id)) \
                             .select("userId", sqlf.col("id").alias("movieId"), "prediction")

    # Exclude already rated
    rated_ids = [r.movieId for r in new_ratings_df.collect()]
    preds = preds.filter(~sqlf.col("movieId").isin(rated_ids))

    return preds.orderBy(sqlf.desc("prediction")).limit(top_n)




if __name__ == "__main__":

  spark = SparkSession.builder.appName("ALS Model Creation").getOrCreate()

  df_ratings_small = get_raw_df(spark, "data/movie-lens-small-latest-dataset/ratings.csv")

  als_model = train_ALS_model(df_ratings_small, "userId", "movieId", "rating")

  # Saving model for later use
  print("\nSaving model...")
  als_model.write().overwrite().save("models/ratings_small_model-latent-features-45")
  print("\n\nModel saved")