from pyspark.sql import SparkSession
import streamlit as st
import pyspark.sql.functions as sqlf
from pyspark.ml.recommendation import ALSModel
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType

from app.als_model import get_new_prediction


if __name__ == "__main__":
  # Formatting movies data
  spark = SparkSession.builder.appName("Movie Recommendation App").getOrCreate()

  df_movies_raw = spark.read.csv("data/movie-lens-small-latest-dataset/movies.csv", header=True, inferSchema=True) 

  df_with_genres = df_movies_raw.withColumn("genreList", sqlf.split(sqlf.col("genres"), "\\|"))

  title_split= sqlf.split(sqlf.col("title"), " \\(")

  df_detailed = df_with_genres.withColumn("movieTitle", title_split.getItem(0)).withColumn("year", title_split.getItem(1))
  df_detailed = df_detailed.withColumn("year", sqlf.regexp_replace("year", "\\)", ""))

  # Streamlit APP
  st.title("Movie Recommender")

  search_text= st.text_input("", "Search...")

  filtered_movies = df_detailed
  if search_text:
      filtered_movies = filtered_movies.filter(
          sqlf.lower(sqlf.col("movieTitle")).like(f"%{search_text.lower()}%")
      )

  results = filtered_movies[['movieId','title', 'movieTitle', 'genreList', 'year']].toPandas()
  results_dict = results[['movieId', 'title']].set_index('movieId').to_dict()['title']
  
  #st.write(results_dict)

  user_movies = dict()

  if "user_movies" not in st.session_state:
    st.session_state["user_movies"] = {}

  def click_on_movie(mov_id, mov_title):
    st.session_state["user_movies"][mov_id] = mov_title

  for k, v in results_dict.items():
    st.button(v, type="primary", use_container_width=False, key = k, on_click=click_on_movie,args=(k, v))

 
  for k, v in results_dict.items():
    st.button(
        v,
        type="primary",
        use_container_width=False,
        key=f"movie_{k}",         # button key must be unique
        on_click=click_on_movie,
        args=(k, v)
    )

st.write("Your selected movies:")
#st.write(st.session_state["user_movies"])

user_ratings = {}
for k, v in st.session_state["user_movies"].items():
  rating_value = st.slider(v, min_value=1.0, max_value=5.0, step =0.5)
  st.write(f"Your rating:", rating_value)
  user_ratings[k] = rating_value


st.write("Your ratings for movies:")
st.write(user_ratings)


def get_new_user_ratings(user_ratings):
  userId = 3000
  user_schema = StructType([
    StructField("movieId", IntegerType(), True),
    StructField("rating", FloatType(), True)
])
  ratings_modified = [{'movieId': k, 'rating': float(v)} for k, v in user_ratings.items()]
  new_user_dataframe = spark.createDataFrame(ratings_modified, schema =user_schema)
  new_user_dataframe.withColumn("userId", sqlf.lit(userId)).show()

  st.write("New user dataframe")
  st.write(new_user_dataframe.toPandas())

  model = ALSModel.load("./models/ratings_small_model-latent-features-45")

    
  new_user_predictions = get_new_prediction(new_user_dataframe, model, new_user_id=3000, top_n=10)

  df_detailed

  new_recommendations = new_user_predictions.join(df_detailed, on="movieId",  how = "inner")
  #new_recommendations.sort("predicted_rating", ascending = [True])

  st.write("New user PRED")
  st.write(new_recommendations.toPandas())

 

st.button("Get Prediction", type="primary", use_container_width = True, on_click= get_new_user_ratings, args=(user_ratings,) )

# TO DO : Implement a button to wipe out information





