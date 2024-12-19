import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

movies = pd.read_csv("dataset/movie.csv")
movies.head()

movies["genres"] = movies["genres"].fillna('')

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])

similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)


def recommend_movies(title, similarity_matrix=similarity_matrix):
    idx = movies[movies["title"] == title].index[0]

    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies["title"].iloc[movie_indices]

print(recommend_movies("Chicken Run (2000)"))