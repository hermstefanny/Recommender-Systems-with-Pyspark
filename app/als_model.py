
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
import pyspark.sql.functions as sqlf
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, LongType
import numpy as np


# Script to train the ALS model

def get_ratings_df(spark_session, file_name,file_schema, separator =','):
    df_raw  = spark_session.read.csv(file_name, sep=separator, header=False, schema = file_schema)
    df = df_raw.select("userId", "movieId", "rating")
    return df


def get_counts(df):
    '''Preliminary Statistics'''
    ratings_counts = df.select("rating").count()
    users_count = df.select("userId").distinct().count()
    movies_count = df.select("movieId").distinct().count()
    ratings_by_userId = df.groupBy("userId").count().show()

    print(f"Total Ratings: {ratings_counts}\n Total user: {users_count}\n Movies count: {movies_count} \nRatings by User: {ratings_by_userId}")


def train_ALS_model(train,  userColumn ="userId", itemColumn = "movieId", ratingColumn = "rating",regParam =0.1, max_iterations=20, latent_factors=10):
 
  als = ALS(
    maxIter=max_iterations,
    regParam=regParam,
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

  
  return model


def test_ALS_model(model, test):
    # Test model
    print("Getting predictions for evaluation....\n")
    predictions = model.transform(test)

    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )

    print("")
    rmse = evaluator.evaluate(predictions)
    print(f"Root-mean-square error on test data = {rmse:.4f}")
  

def fine_tune_ALS(train_set, validation_set, latent_factors, regParams, maxIter=15):
    ''' Function to fine tune ALS model and get the best model with optimized parameters'''
    
    print("Building model...")
    als = ALS(userCol = "userId", itemCol = "movieId", ratingCol = "rating", coldStartStrategy = "drop", maxIter = maxIter)
    
    print("Creating the parameter grid...")
    paramGrid = ParamGridBuilder().addGrid(als.rank, latent_factors).addGrid(als.regParam, regParams).build()

    print("Appplying the evaluator...")
    evaluator = RegressionEvaluator(metricName = "rmse", labelCol = "rating", predictionCol = "prediction")

    print("Initializing cross validation...")
    cv = CrossValidator(estimator = als, estimatorParamMaps = paramGrid, evaluator = evaluator, numFolds = 3)

    cv_model = cv.fit(train_set)

    print("Evaluating on validation set...")
    predictions = cv_model.bestModel.transform(validation_set)
    rmse_val = evaluator.evaluate(predictions)

    # Parameters for the best model
    best_rank = cv_model.bestModel.rank
    best_reg = cv_model.bestModel._java_obj.parent().getRegParam()
    best_maxIter = cv_model.bestModel._java_obj.parent().getMaxIter()

    print(f"\nBest ALS parameters → rank={best_rank}, regParam={best_reg}, maxIter={best_maxIter}")
    print(f"\nBest model RMSE on validation data: {rmse_val}")

    return cv_model.bestModel


def get_cold_start_prediction(new_user_ratings_df, model, new_user_id=100001, top_n=10):
    '''
    Cold-start ALS: recommend for a new user without retraining.
    '''
    # Join new ratings with item factors
    rated = new_user_ratings_df.join(model.itemFactors,
                                  new_user_ratings_df.movieId == model.itemFactors.id,
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
    rated_ids = [r.movieId for r in new_user_ratings_df.collect()]
    preds = preds.filter(~sqlf.col("movieId").isin(rated_ids))

    return preds.orderBy(sqlf.desc("prediction")).limit(top_n)



def get_retrained_prediction(ratings_with_new_user, new_userId):

    new_model = train_ALS_model(ratings_with_new_user)

    ratings_with_new_user.filter(ratings_with_new_user.userId ==new_userId).show()

    userRecs = new_model.recommendForAllUsers(10)
    new_UserRec= userRecs.filter(userRecs.userId == new_userId)

    flat_new_UserRec = new_UserRec.withColumn("rec", sqlf.explode(sqlf.col("recommendations"))) \
    .select(
        sqlf.col("userId"),
        sqlf.col("rec.movieId").alias("movieId"),
        sqlf.col("rec.rating").alias("predicted_rating")
    )

    rated_ids = (
        ratings_with_new_user
        .filter(sqlf.col("userId") == new_userId)
        .select("movieId")
        .rdd.flatMap(lambda x: x)
        .collect()
    )

    final_predictions = flat_new_UserRec.filter(~sqlf.col("movieId").isin(rated_ids))
    
    final_predictions.show()

    return final_predictions



if __name__ == "__main__":

  spark = SparkSession.builder.appName("ALS Model Creation")\
  .config("spark.driver.memory", "8g")\
  .config("spark.executor.memory", "8g")\
  .config("spark.driver.max.ResultSize", "8g")\
  .getOrCreate()

  ratings_schema = StructType([
    StructField("userId", IntegerType(), True),
    StructField("movieId", IntegerType(), True),
    StructField("rating", FloatType(), True),
    StructField("timestamp", LongType(), True),
  ])

  df_ratings = get_ratings_df(spark, "data/movielens-1m-dataset/ratings.dat", ratings_schema,'::' )

  
  (train, test) = df_ratings.randomSplit([0.8, 0.2], seed=42)

  # Model without cross-validation
  #als_model.write().overwrite().save("models/ratings_model-latent-features-45")

  #als_model = train_ALS_model(train, test, "userId", "movieId", "rating")


  # Optimized Model
  (train_set, validation_set) = train.randomSplit([0.8, 0.2], seed=30)

  latent_factors = [10, 15, 20, 25, 30, 35, 40, 45, 50]
  regParams = [0.05, 0.1, 0.15]
  optimized_als_model= fine_tune_ALS(train_set, validation_set, latent_factors, regParams)

  test_ALS_model(optimized_als_model, test)

  print("\nSaving model...")

  optimized_als_model.write().overwrite().save("models/ratings_model-optimized")
  print("\n\nModel saved")