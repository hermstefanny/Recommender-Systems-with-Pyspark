FROM apache/spark:4.0.1-scala2.13-java17-python3-r-ubuntu
    
USER root
COPY requirements.txt .
RUN pip3 install -r requirements.txt
USER spark
WORKDIR /app

COPY app/ ./app
COPY data/movielens-1m-dataset ./data/movielens-1m-dataset
COPY models/ratings_model-optimized ./models/ratings_model-optimized

EXPOSE 8502

HEALTHCHECK CMD curl --fail http://localhost:8502/_stcore/health

ENV PYTHONPATH=/app

ENTRYPOINT ["streamlit", "run", "app/recommendation_ui.py", "--server.port=8502", "--server.address=0.0.0.0"]