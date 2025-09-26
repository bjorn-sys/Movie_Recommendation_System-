# ========================================================================
# Movie Recommendation System using TF-IDF + Cosine Similarity
# Dataset Columns: MOVIES | YEAR | GENRE | RATING | ONE-LINE | STARS | VOTES | RunTime | Gross
# ========================================================================

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# -----------------------------------------------------------------------------
# 1. Load and preprocess data
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")   # <-- replace with your actual file name/path
    
    # Rename columns to standard names
    df = df.rename(columns={
        "MOVIES": "title",
        "GENRE": "genres",
        "ONE-LINE": "overview"
    })
    
    # Fill missing values
    df["genres"] = df["genres"].fillna("")
    df["overview"] = df["overview"].fillna("")
    
    # Combine text features for TF-IDF
    df["combined_features"] = df["genres"] + " " + df["overview"]
    
    # TF-IDF vectorization
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["combined_features"])
    
    # Cosine similarity
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    # Map titles to index
    indices = pd.Series(df.index, index=df["title"]).drop_duplicates()
    
    return df, cosine_sim, indices

# -----------------------------------------------------------------------------
# 2. Recommendation Function
# -----------------------------------------------------------------------------
def recommend_movies(title, n=5, genre_filter=None):
    if title not in indices:
        similar_titles = [t for t in indices.index if title.lower() in t.lower()]
        if similar_titles:
            return f"❗ Movie '{title}' not found. Did you mean: {', '.join(similar_titles[:3])}?"
        else:
            return f"❗ Movie '{title}' not found in the dataset."
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Skip the first one (itself)
    sim_scores = sim_scores[1:n+6]
    
    movie_indices = [i[0] for i in sim_scores]
    similarity_values = [i[1] for i in sim_scores]
    
    recommendations = df[["title", "genres", "YEAR", "RATING"]].iloc[movie_indices].copy()
    recommendations["Similarity_Score"] = similarity_values
    
    if genre_filter:
        recommendations = recommendations[
            recommendations["genres"].str.lower().str.contains(genre_filter.lower())
        ]
    
    return recommendations.head(n)

# -----------------------------------------------------------------------------
# 3. Streamlit App UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

st.title("🎬 Movie Recommendation System")
st.markdown("Get similar movie suggestions based on your favorite movie!")

# Load data
df, cosine_sim, indices = load_data()

# Sidebar
st.sidebar.header("🔎 Find Recommendations")
movie_list = df["title"].tolist()
selected_movie = st.sidebar.selectbox("Select a movie you like:", movie_list)

genre_option = st.sidebar.text_input("Optional: Filter by genre (e.g., Action, Comedy)")
n_recommend = st.sidebar.slider("Number of recommendations:", 1, 20, 5)

# Recommend Button
if st.sidebar.button("Recommend"):
    results = recommend_movies(selected_movie, n_recommend, genre_option)
    
    if isinstance(results, str):
        st.warning(results)
    else:
        st.success(f"✅ Here are your top {len(results)} recommendations similar to **{selected_movie}**:")
        st.dataframe(results.reset_index(drop=True))

# Show raw data
with st.expander("📂 View Dataset"):
    st.write(df.head(20))
