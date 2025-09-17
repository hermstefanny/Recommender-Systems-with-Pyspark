from pyspark.sql import SparkSession
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator




def get_raw_df(spark_session, file_name):
  df_raw  = spark_session.read.csv(file_name, header=True, inferSchema=True)
  return df_raw


# Preliminary Statistics
ratings_counts = df_ratings_small.select("rating").count()
users_count = df_ratings_small.select("userId").distinct().count()
movies_count = df_ratings_small.select("movieId").distinct().count()
ratings_by_usetId = df_ratings_small.groupBy("userId").count().show()

print(f"Total Ratings: {ratings_counts}\n Total user: {users_count}\n Movies count: {movies_count}")



(train, test) = df_ratings_small.randomSplit([0.8, 0.2], seed=42)


if __name__ == "__main__":
  spark = SparkSession.builder.appName("Movie Recommendation App").getOrCreate()
  df_ratings_small = get_raw_df(spark, "data/movie-lens-small-latest-dataset/ratings.csv")