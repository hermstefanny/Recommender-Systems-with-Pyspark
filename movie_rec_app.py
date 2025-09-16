from pyspark.sql import SparkSession


spark = SparkSession.builder.appName("Movie Recommendation App").getOrCreate()


# Reading the ratings_small csv file

df_ratings_small  = spark.read.csv("ratings_small.csv", header=True, inferSchema=True)

df_ratings_small.show()

#counting nu
ratings_counts = df_ratings_small.select("rating").count()
users_count = df_ratings_small.select("userId").distinct().count()
movies_count = df_ratings_small.select("movieId").distinct().count()