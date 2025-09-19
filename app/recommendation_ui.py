from pyspark.sql import SparkSession
import pandas as pd
import streamlit as st
import pyspark.sql.functions as sqlf
from streamlit_card import card
from pyspark.ml.recommendation import ALSModel
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, LongType

from app.als_model import get_cold_start_prediction, get_retrained_prediction


def new_user_df(user_ratings):
    userId = 100001
    user_schema = StructType([
    StructField("movieId", IntegerType(), True),
    StructField("rating", FloatType(), True)
])
    ratings_modified = [{'movieId': k, 'rating': float(v)} for k, v in user_ratings.items()]
    new_user_dataframe = spark.createDataFrame(ratings_modified, schema =user_schema)
    new_user_dataframe = new_user_dataframe.withColumn("userId", sqlf.lit(userId))

    
    return new_user_dataframe


def get_cold_start_ratings(user_ratings, df_detailed):
  new_user_dataframe = new_user_df(user_ratings)

  # st.write("New user dataframe")
  # st.write(new_user_dataframe.toPandas())

  # Predictions
  model = ALSModel.load("./models/ratings_model-optimized")

  ## Cold Start User Predictions
  new_user_cold_predictions = get_cold_start_prediction(new_user_dataframe.select("userId", "movieId", "rating"), model, new_user_id=100001, top_n=5)

  new_recommendations = new_user_cold_predictions.join(df_detailed, on="movieId",  how = "inner")

  return new_recommendations.toPandas()


def get_retrained_ratings(user_ratings, df_detailed, df_ratings_raw):

  new_user_dataframe = new_user_df(user_ratings)
  ratings_with_new_u = df_ratings_raw.select("userId", "movieId", "rating").union(new_user_dataframe.select("userId", "movieId", "rating"))
  new_user_ret_predictions= get_retrained_prediction(ratings_with_new_u, new_userId=100001)

  new_recommendations = new_user_ret_predictions.join(df_detailed, on="movieId",  how = "inner")
  
  return new_recommendations.toPandas()
  

def handle_data(spark):
  
    movies_schema = StructType([
    StructField("movieId", IntegerType(), True),
    StructField("title", StringType(), True),
    StructField("genres", StringType(), True),
  ])

    ratings_schema = StructType([
    StructField("userId", IntegerType(), True),
    StructField("movieId", IntegerType(), True),
    StructField("rating", FloatType(), True),
    StructField("timestamp", LongType(), True),
  ])

    df_movies_raw = spark.read.csv("data/movielens-1m-dataset/movies.dat",  sep='::', header=False, schema = movies_schema)
    df_ratings_raw = spark.read.csv("data/movielens-1m-dataset/ratings.dat",  sep='::', header=False, schema = ratings_schema) 

    df_with_genres = df_movies_raw.withColumn("genreList", sqlf.split(sqlf.col("genres"), "\\|"))

    title_split= sqlf.split(sqlf.col("title"), " \\(")

    df_detailed = df_with_genres.withColumn("movieTitle", title_split.getItem(0)).withColumn("year", title_split.getItem(1))
    df_detailed = df_detailed.withColumn("year", sqlf.regexp_replace("year", "\\)", ""))

    return df_detailed, df_ratings_raw

def click_on_movie(mov_id, mov_title):
    st.session_state["user_movies"][mov_id] = mov_title


if __name__ == "__main__":

  spark = SparkSession.builder.appName("Movie Recommendation App").getOrCreate()
  # Formatting movies data
  (df_detailed, df_ratings_raw) = handle_data(spark)

  # Streamlit APP
  st.title("🍿 Movie Recommender 🍿")
  left_column, center_column, right_column = st.columns(3)

  with left_column:
    st.subheader ("Your search starts here ")
    search_text= st.text_input("", "Search...")

    if "search_text" not in st.session_state:
      st.session_state["search_text"] = ""

    if search_text.strip():
        filtered_movies = df_detailed.filter(
            sqlf.lower(sqlf.col("movieTitle")).like(f"%{search_text.lower()}%")
        )
        st.session_state["filtered_movies"] = filtered_movies
    else:
       st.session_state["filtered_movies"] = None

    results=pd.DataFrame()
    if "results" not in st.session_state:
      st.session_state["results"] = pd.DataFrame()
    
    results = filtered_movies[['movieId','title', 'movieTitle', 'genreList', 'year']].toPandas()
    results_dict = results[['movieId', 'title']].set_index('movieId').to_dict()['title']

    if "results_dict" not in st.session_state:
      st.session_state["results_dict"] = {}

    for k, v in results_dict.items():
      st.button(
          v,
          type="secondary",
          use_container_width=False,
          key=f"movie_{k}",         
          on_click=click_on_movie,
          args=(k, v)
      )
  
    user_movies = dict()
    if "user_movies" not in st.session_state:
      st.session_state["user_movies"] = {}

with center_column:
  st.subheader("Give a 5-star rating ⭐")

  user_ratings = {}
  for k, v in st.session_state["user_movies"].items():
    rating_value = st.slider(v, min_value=1, max_value=5, step =1)
    st.write(f"Your rating:", rating_value)
    user_ratings[k] = rating_value


with right_column:
  st.subheader("Get your prediction 🔮")

  recs_df = pd.DataFrame()
  if st.button(
      "Get Cold Start Prediction",
      type="primary",
      use_container_width=True
  ):
      recs_df = get_cold_start_ratings(user_ratings, df_detailed)
  
  #st.button("Get Cold Start Prediction", type="primary", use_container_width = True, on_click = get_cold_start_ratings, args=(user_ratings,df_detailed,) )

  #st.button("Get Retrained Prediction", type="primary", use_container_width = True,  on_click = get_retrained_ratings, args=(user_ratings,df_detailed, df_ratings_raw,) )

  retrained_df = pd.DataFrame()
  if st.button(
      "Get Retrained Prediction",
      type="primary",
      use_container_width=True
  ):
      recs_df = get_retrained_ratings(user_ratings, df_detailed, df_ratings_raw)


output_area = st.container()

if st.button("Other chance",
             type="primary",
             use_container_width = True
             ): 
    st.session_state.clear()   # wipes all state
    st.rerun()     

with output_area:
  st.subheader(" 🎬 What you should watch next 🎬")

  # for index, row in retrained_df.iterrows():
  #   card(
  #       title=row["movieTitle"],
  #       text = row['genres'])

  cols = st.columns(5)  
  for i, (_, row) in enumerate(recs_df.iterrows()):
      with cols[i % 5]:
          st.markdown(f"**{row['movieTitle']}**  \n<span style='color:gray'>{row['genres']}</span>", unsafe_allow_html=True)
      if (i + 1) % 5 == 0:  
          cols = st.columns(5)


# TO DO : Implement a button to wipe out information