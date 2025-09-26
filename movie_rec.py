# ================================================================
# MOVIE RECOMMENDER STREAMLIT APP
# ================================================================

# -----------------
# 1. IMPORT LIBRARIES
# -----------------
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------
# 2. LOAD DATA
# -----------------
# Replace with your dataset path
df = pd.read_csv("movies.csv")       # must have columns: MOVIES, GENRE, DESCRIPTION
df = df.dropna(subset=['MOVIES', 'DESCRIPTION'])

# -----------------
# 3. CREATE FEATURES AND COSINE SIMILARITY MATRIX
# -----------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['DESCRIPTION'])

cosine_sim = cosine_similarity(tfidf_matrix)   # (num_movies x num_movies)

# Map movie titles to their indices
indices = pd.Series(df.index, index=df['MOVIES']).drop_duplicates()

# -----------------
# 4. SESSION STATE INITIALIZATION
# -----------------
if "get_recs" not in st.session_state:
    st.session_state.get_recs = False

if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None

if "selected_genre" not in st.session_state:
    st.session_state.selected_genre = None

if "num_recs" not in st.session_state:
    st.session_state.num_recs = 5

# -----------------
# 5. RECOMMENDATION FUNCTION
# -----------------
def recommend_movies(movie_title, n=5, genre_filter=None):
    # Check if the movie exists
    if movie_title not in indices:
        similar_titles = [title for title in indices.index if movie_title.lower() in title.lower()]
        if similar_titles:
            return f"❗ Movie '{movie_title}' not found. Did you mean: {', '.join(similar_titles[:3])}?"
        else:
            return f"❗ Movie '{movie_title}' not found in dataset!"

    # Get index of the selected movie
    idx = indices[movie_title]

    # Compute similarity scores
    sim_scores = list(enumerate(cosine_sim[idx].flatten()))

    # Sort movies by similarity score
    sim_scores = sorted(sim_scores, key=lambda x: float(x[1]), reverse=True)

    # Take top n+5 (skip the movie itself)
    sim_scores = sim_scores[1:n+6]

    # Extract movie indices
    movie_indices = [i[0] for i in sim_scores]
    similarity_values = [i[1] for i in sim_scores]

    # Create recommendations DataFrame
    recommendations = df[['MOVIES', 'GENRE']].iloc[movie_indices].copy()
    recommendations['Similarity_Score'] = similarity_values

    # Apply genre filter if provided
    if genre_filter:
        recommendations = recommendations[
            recommendations['GENRE'].str.lower().str.contains(genre_filter.lower())
        ]

    return recommendations.head(n)

# -----------------
# 6. STREAMLIT UI
# -----------------
st.title("🎬 Movie Recommendation App")
st.markdown("Find movies similar to your favorites!")

# Movie selection
st.session_state.selected_movie = st.selectbox(
    "Select or type a movie title:",
    df['MOVIES'].unique()
)

# Genre filter (optional)
st.session_state.selected_genre = st.text_input("Optional: Filter by genre (e.g., Action, Comedy):")

# Number of recommendations
st.session_state.num_recs = st.slider(
    "Number of recommendations:",
    min_value=1,
    max_value=20,
    value=5
)

# Recommendation button
if st.button("🔎 Get Recommendations"):
    st.session_state.get_recs = True

# Display results
if st.session_state.get_recs:
    results = recommend_movies(
        st.session_state.selected_movie,
        st.session_state.num_recs,
        st.session_state.selected_genre
    )

    if isinstance(results, str):  # if error message
        st.warning(results)
    else:
        st.success("✅ Recommendations:")
        st.dataframe(results, use_container_width=True)
