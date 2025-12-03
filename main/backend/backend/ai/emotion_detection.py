"""Emotion detection and movie recommendation utilities for Vyber.

This module exposes three main helpers for the FastAPI backend:

- load_movies()        → returns the movies DataFrame with ratings and genres
- detect_mood(text)    → maps free-text input to one of our moods
- recommend(mood, ...) → returns a list of recommended movie titles with explanations
"""

import os
import ast
import numpy as np
import pandas as pd
from transformers import pipeline
import joblib

# --- Load precomputed artifacts (vectorizer, similarity matrix, movies) ---

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
MODELS_DIR = os.path.abspath(MODELS_DIR)

TFIDF_VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
COSINE_SIM_MATRIX_PATH = os.path.join(MODELS_DIR, "cosine_sim_matrix.npy")
MOVIES_DF_PATH = os.path.join(MODELS_DIR, "loaded_movies_df.csv")

# Load TF-IDF vectorizer (kept for future use)
tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)

# Load cosine similarity matrix
cosine_sim_matrix = np.load(COSINE_SIM_MATRIX_PATH)

# Load movies with ratings
movies_df = pd.read_csv(MOVIES_DF_PATH)

# Ensure genres are a proper Python list
def _ensure_genres_list(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str) and val.startswith("[") and "]" in val:
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    if isinstance(val, str):
        return val.split("|")
    return []

if "genres" in movies_df.columns:
    movies_df["genres"] = movies_df["genres"].apply(_ensure_genres_list)

# --- Emotion model and mapping ---

# Pretrained HuggingFace model for emotion classification
emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)

# Map fine-grained emotions → 6 Vyber moods
emotion_to_mood_map = {
    "joy": "happy",
    "optimism": "happy",
    "admiration": "happy",
    "amusement": "happy",
    "surprise": "happy",
    "trust": "happy",
    "contentment": "happy",
    "love": "romantic",
    "caring": "romantic",
    "sadness": "sad",
    "grief": "sad",
    "disappointment": "sad",
    "anger": "action",
    "annoyance": "action",
    "disgust": "scary",
    "fear": "scary",
    "nervousness": "scary",
    "anticipation": "fantasy",
    "curiosity": "fantasy",
    "excitement": "fantasy",
}

# Fallback if nothing matches
DEFAULT_MOOD = "happy"

# Mood → genres mapping (optimized)
mood_to_genres_map = {
    "happy": [
        "Comedy",
        "Family",
        "Animation",
        "Romance"
    ],
    "sad": [
        "Drama",
        "Romance"
    ],
    "romantic": [
        "Romance",
        "Drama"
    ],
    "action": [
        "Action",
        "Adventure",
        "Crime",
        "Sci-Fi"
    ],
    "scary": [
        "Horror",
        "Thriller"
    ],
    "fantasy": [
        "Fantasy",
        "Sci-Fi",
        "Animation",
        "Adventure"
    ]
}

def load_movies():
    """Return the full movies DataFrame used by the recommender."""
    return movies_df.copy()

def _extract_top_dict(results):
    """Safely pull out the top {label, score} dict from any nested list structure."""
    obj = results
    # Sometimes it's already a dict
    if isinstance(obj, dict):
        return obj
    # If it's a list, keep going into the first element until we hit a dict or fail
    while isinstance(obj, list) and len(obj) > 0:
        obj = obj[0]
        if isinstance(obj, dict):
            return obj
    return None

def detect_mood(text: str) -> str:
    """Detect a coarse mood (one of 6) from free-text input.

    Handles different output shapes from the HuggingFace pipeline.
    If anything goes wrong, returns DEFAULT_MOOD.
    """
    if not isinstance(text, str) or not text.strip():
        return DEFAULT_MOOD

    try:
        results = emotion_pipeline(text)
    except Exception:
        return DEFAULT_MOOD

    top = _extract_top_dict(results)
    if top is None:
        return DEFAULT_MOOD

    label = str(top.get("label", "")).lower()
    mood = emotion_to_mood_map.get(label, DEFAULT_MOOD)
    return mood

def recommend(mood: str, top_n: int = 5, weight_sim: float = 0.7, weight_rating: float = 0.3):
    """Recommend movies for a given mood.

    Combines cosine similarity (based on title + genres)
    and average rating to score movies, then returns a list
    of dicts with title, genres, mood, rating, and explanation.
    """
    mood = (mood or "").lower()
    if mood not in mood_to_genres_map:
        mood = DEFAULT_MOOD

    target_genres = mood_to_genres_map[mood]

    # Simple genre filter: keep movies that contain at least one target genre
    def has_genre(genres):
        if not isinstance(genres, (list, tuple, set)):
            return False
        genres_lower = [str(g).lower() for g in genres]
        return any(tg.lower() in genres_lower for tg in target_genres)

    mask = movies_df["genres"].apply(has_genre)
    candidate_indices = movies_df.index[mask].tolist()

    if not candidate_indices:
        # Fallback: if no movie matches, just take all movies
        candidate_indices = list(movies_df.index)

    # Slice similarity matrix & ratings for the candidates
    sim_submatrix = cosine_sim_matrix[np.ix_(candidate_indices, candidate_indices)]

    # For simplicity, use the average similarity of each candidate to all others
    sim_scores = sim_submatrix.mean(axis=1)

    # Use avg_rating column if present, else fallback to ones
    if "avg_rating" in movies_df.columns:
        ratings = movies_df.loc[candidate_indices, "avg_rating"].values
    else:
        ratings = np.ones(len(candidate_indices))

    # Normalize scores to [0,1] to combine them
    def _normalize(x):
        x = np.asarray(x, dtype=float)
        if x.max() == x.min():
            return np.ones_like(x)
        return (x - x.min()) / (x.max() - x.min())

    sim_norm = _normalize(sim_scores)
    rating_norm = _normalize(ratings)

    final_scores = weight_sim * sim_norm + weight_rating * rating_norm

    # Sort candidates by score
    sorted_idx = np.argsort(final_scores)[::-1]  # descending
    top_idx = sorted_idx[:top_n]

    top_movie_indices = [candidate_indices[i] for i in top_idx]
    top_movies = movies_df.loc[top_movie_indices]

    # Build rich output: title, genres, mood, rating, explanation
    results = []
    for _, row in top_movies.iterrows():
        genres_val = row.get("genres", [])
        # Make sure genres is a list for JSON
        if isinstance(genres_val, str):
            try:
                parsed = ast.literal_eval(genres_val)
                if isinstance(parsed, list):
                    genres_val = parsed
                else:
                    genres_val = [genres_val]
            except Exception:
                genres_val = [genres_val]
        elif not isinstance(genres_val, (list, tuple, set)):
            genres_val = [str(genres_val)]

        genres_list = [str(g) for g in genres_val if g is not None]
        main_genre = genres_list[0] if genres_list else None

        if main_genre:
            explanation = (
                f"Because you're feeling {mood}, we picked this {main_genre} movie "
                f"that matches your current vibe."
            )
        else:
            explanation = (
                f"Because you're feeling {mood}, we picked this highly-rated movie "
                f"that many people enjoy in this mood."
            )

        avg_rating = None
        if "avg_rating" in row and not pd.isna(row["avg_rating"]):
            avg_rating = float(row["avg_rating"])

        results.append({
            "title": row["title"],
            "genres": genres_list,
            "mood": mood,
            "avg_rating": avg_rating,
            "explanation": explanation
        })

    return results
