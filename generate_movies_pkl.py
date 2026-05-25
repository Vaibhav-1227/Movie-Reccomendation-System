import pandas as pd
import pickle
import ast

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

movies = movies.merge(credits, on="title")
movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
movies.dropna(inplace=True)

def convert(obj):
    return [i["name"] for i in ast.literal_eval(obj)]

def convert3(obj):
    return [i["name"] for i in ast.literal_eval(obj)[:3]]

def fetch_director(obj):
    return [i["name"] for i in ast.literal_eval(obj) if i["job"] == "Director"]

movies["genres"]   = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)
movies["cast"]     = movies["cast"].apply(convert3)
movies["crew"]     = movies["crew"].apply(fetch_director)
movies["overview"] = movies["overview"].apply(lambda x: x.split())

for col in ["genres", "keywords", "cast", "crew"]:
    movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])

movies["tags"] = movies["overview"] + movies["genres"] + movies["keywords"] + movies["cast"] + movies["crew"]

new_df = movies[["movie_id", "title", "tags"]].copy()
new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x).lower())

pickle.dump(new_df.to_dict(), open("movies.pkl", "wb"))
print("✅ movies.pkl generated!")