

import ast
import pickle
import numpy as np
import pandas as pd

# text processing
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ensure NLTK resources (porter stemmer doesn't need downloads; this is safe)
# nltk.download('punkt')  # uncomment if you plan to use tokenizers that need it

ps = PorterStemmer()

# ---------- Helper functions ----------
def safe_literal_eval(x):
    """Safely evaluate JSON-like strings to Python objects. Return [] on failure."""
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        # some rows may already be Python lists or malformed — fall back gracefully
        return []

def get_list_of_names(json_str):
    """
    Given a JSON-style list of dicts (as string or list), return list of names.
    Handles None/NaN and malformed values gracefully.
    """
    items = safe_literal_eval(json_str)
    names = []
    for it in items:
        try:
            # Expect dict with 'name' key
            name = it.get('name') if isinstance(it, dict) else None
            if name:
                names.append(str(name))
        except Exception:
            continue
    return names

def get_top_cast_names(json_str, top_n=3):
    """Return up to top_n cast member names from the cast field."""
    items = safe_literal_eval(json_str)
    names = []
    for i, it in enumerate(items):
        if i >= top_n:
            break
        try:
            name = it.get('name') if isinstance(it, dict) else None
            if name:
                names.append(str(name))
        except Exception:
            continue
    return names

def get_director_name(json_str):
    """Return a single-director name (if present) as a list (to keep consistent structure)."""
    items = safe_literal_eval(json_str)
    for it in items:
        try:
            if isinstance(it, dict) and it.get('job') == 'Director':
                name = it.get('name')
                if name:
                    return [str(name)]
        except Exception:
            continue
    return []

def remove_spaces_from_list_elements(lst):
    """Replace spaces inside list elements (e.g., 'Tom Cruise' -> 'TomCruise')"""
    return [str(x).replace(" ", "") for x in lst]

def stem_text(text):
    """Stem every word in the input string using PorterStemmer."""
    if not isinstance(text, str):
        text = str(text)
    return " ".join(ps.stem(word) for word in text.split())

# ---------- Load CSVs
movies_csv = "tmdb_5000_movies.csv"
credits_csv = "tmdb_5000_credits.csv"

try:
    movies = pd.read_csv(movies_csv)
    credits = pd.read_csv(credits_csv)
except FileNotFoundError as e:
    raise SystemExit(f"Missing dataset file: {e}. Place '{movies_csv}' and '{credits_csv}' in the working directory.")

# ---------- Merge & select relevant columns ----------
# Merge on title (as original). If you prefer merging by id change logic accordingly.
merged = movies.merge(credits, on='title', how='inner')

# Keep only relevant columns; ensure names match your downstream expectations
expected_cols = ['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']
for col in expected_cols:
    if col not in merged.columns:
        raise SystemExit(f"Required column '{col}' not found after merge. Check your CSVs.")

merged = merged[expected_cols].copy()

# ---------- Clean & preprocess ----------
# Drop rows where essential fields are missing (movie_id or title or overview) but be explicit
merged.dropna(subset=['movie_id', 'title', 'overview'], inplace=True)
merged.reset_index(drop=True, inplace=True)

# Convert movie_id to int if possible
def safe_int(x):
    try:
        return int(x)
    except Exception:
        return np.nan

merged['movie_id'] = merged['movie_id'].apply(safe_int)
merged.dropna(subset=['movie_id'], inplace=True)
merged['movie_id'] = merged['movie_id'].astype(int)
merged.reset_index(drop=True, inplace=True)

# Parse JSON-like columns into lists of strings (names)
merged['genres'] = merged['genres'].apply(get_list_of_names)
merged['keywords'] = merged['keywords'].apply(get_list_of_names)
merged['cast'] = merged['cast'].apply(lambda x: get_top_cast_names(x, top_n=3))
merged['crew'] = merged['crew'].apply(get_director_name)

# Tokenize overview into list of words (simple split)
merged['overview'] = merged['overview'].fillna("").apply(lambda x: str(x).split())

# Remove spaces within multi-word names so they behave like single tokens
merged['genres'] = merged['genres'].apply(remove_spaces_from_list_elements)
merged['keywords'] = merged['keywords'].apply(remove_spaces_from_list_elements)
merged['cast'] = merged['cast'].apply(remove_spaces_from_list_elements)
merged['crew'] = merged['crew'].apply(remove_spaces_from_list_elements)

# Create tags by concatenating all token lists
merged['tags'] = merged['overview'] + merged['genres'] + merged['keywords'] + merged['cast'] + merged['crew']

# Build new dataframe with the required fields and produce string tags
new_df = merged[['movie_id', 'title', 'tags']].copy()
# join tags to single string and lowercase
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

# Optional: remove duplicate titles (keep first)
new_df.drop_duplicates(subset=['title'], keep='first', inplace=True)
new_df.reset_index(drop=True, inplace=True)

# ---------- Stemming ----------
new_df['tags'] = new_df['tags'].apply(stem_text)

# ---------- Vectorization ----------
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# ---------- Similarity matrix ----------
similarity = cosine_similarity(vectors)

# ---------- Sanity checks & prints ----------
print("Processed movies:", new_df.shape)
print("Example row (first):")
print(new_df.head(1).T)

# ---------- Persist outputs ----------
# 1) Save the full DataFrame for other uses
pickle.dump(new_df, open('movies.pkl', 'wb'))

# 2) Save a dictionary form (like your Streamlit app expects)
movie_dict = new_df.to_dict(orient='list')  # keys -> list of values
# some older code expects movie_dict to map index -> row dict; if so, use:
# movie_dict_indexed = new_df.to_dict(orient='index')
pickle.dump(movie_dict, open('movie_dict.pkl', 'wb'))

# 3) Save similarity matrix (numpy array)
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("Saved: movies.pkl, movie_dict.pkl, similarity.pkl")
