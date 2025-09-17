from pyspark.sql import SparkSession
import streamlit as st
import pyspark.sql.functions as sqlf

from streamlit_card import card


if __name__ == "__main__":
  # Working to refine the movies data
  spark = SparkSession.builder.appName("Movie Recommendation App").getOrCreate()

  df_movies_raw = spark.read.csv("data/movie-lens-small-latest-dataset/movies.csv", header=True, inferSchema=True) 

  df_with_genres = df_movies_raw.withColumn("genreList", sqlf.split(sqlf.col("genres"), "\\|"))

  title_split= sqlf.split(sqlf.col("title"), " \\(")

  df_detailed = df_with_genres.withColumn("movieTitle", title_split.getItem(0)).withColumn("year", title_split.getItem(1))
  df_detailed = df_detailed.withColumn("year", sqlf.regexp_replace("year", "\\)", ""))

  df_detailed.show(5)

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
  print(results_dict)
  
  st.write(results_dict)

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
st.write(st.session_state["user_movies"])

user_ratings = {}
for k, v in st.session_state["user_movies"].items():
  rating_value = st.slider(v, min_value=1, max_value=5, step =1)
  st.write(f"Your rating:", rating_value)
  user_ratings[k] = rating_value


st.write("Your ratings for movies:")
st.write(user_ratings)



