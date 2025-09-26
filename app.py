import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .movie-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'get_recs' not in st.session_state:
    st.session_state.get_recs = False
if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = ""
if 'num_recs' not in st.session_state:
    st.session_state.num_recs = 5
if 'selected_genre' not in st.session_state:
    st.session_state.selected_genre = ""

# Title
st.markdown('<h1 class="main-header">🎬 Movie Recommendation System</h1>', unsafe_allow_html=True)

# Sample movie data (replace this with your actual dataset)
@st.cache_data
def load_sample_data():
    """Load sample movie data"""
    movies = {
        'title': [
            'The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 
            'Pulp Fiction', 'Forrest Gump', 'Inception', 'The Matrix',
            'Goodfellas', 'The Silence of the Lambs', 'Star Wars: A New Hope',
            'The Lord of the Rings: The Fellowship of the Ring', 'Fight Club',
            'Parasite', 'Spirited Away', 'Interstellar'
        ],
        'genres': [
            'Drama', 'Crime,Drama', 'Action,Crime,Drama', 'Crime,Drama',
            'Drama,Romance', 'Action,Adventure,Sci-Fi', 'Action,Sci-Fi',
            'Biography,Crime,Drama', 'Crime,Drama,Thriller', 'Action,Adventure,Fantasy',
            'Action,Adventure,Drama', 'Drama', 'Comedy,Drama,Thriller',
            'Animation,Adventure,Family', 'Adventure,Drama,Sci-Fi'
        ],
        'overview': [
            'Two imprisoned men bond over a number of years...',
            'The aging patriarch of an organized crime dynasty...',
            'When the menace known as the Joker wreaks havoc...',
            'The lives of two mob hitmen, a boxer, and a pair of diner bandits...',
            'The presidencies of Kennedy and Johnson...',
            'A thief who steals corporate secrets...',
            'A computer hacker learns from mysterious rebels...',
            'The story of Henry Hill and his life in the mob...',
            'A young F.B.I. cadet must receive the help of an incarcerated...',
            'Luke Skywalker joins forces with a Jedi Knight...',
            'A meek Hobbit from the Shire and eight companions...',
            'An insomniac office worker and a devil-may-care soapmaker...',
            'Greed and class discrimination threaten the newly formed...',
            'During her family\'s move to the suburbs, a sullen 10-year-old...',
            'A team of explorers travel through a wormhole in space...'
        ]
    }
    return pd.DataFrame(movies)

# Load data
df = load_sample_data()

# Create features for recommendation
df['combined_features'] = df['title'] + ' ' + df['genres'] + ' ' + df['overview']

# Create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create indices for movie titles
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def recommend_movies(movie_title, n=5, genre_filter=None):
    """Get movie recommendations based on cosine similarity"""
    try:
        # Get the index of the movie
        idx = indices[movie_title]
        
        # Get similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Sort movies by similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top n+1 movies (including the input movie)
        sim_scores = sim_scores[1:n+6]
        
        # Get movie indices
        movie_indices = [i[0] for i in sim_scores]
        
        # Get movie details
        recommendations = df.iloc[movie_indices][['title', 'genres', 'overview']].copy()
        recommendations['similarity_score'] = [i[1] for i in sim_scores]
        
        # Filter by genre if specified
        if genre_filter and genre_filter != "All":
            recommendations = recommendations[
                recommendations['genres'].str.contains(genre_filter, case=False, na=False)
            ]
        
        return recommendations.head(n)
    
    except KeyError:
        st.error(f"Movie '{movie_title}' not found in the database.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return pd.DataFrame()

# Main layout
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<h3 class="sub-header">🎯 Find Similar Movies</h3>', unsafe_allow_html=True)
    
    # Movie selection
    movie_titles = sorted(df['title'].tolist())
    selected_movie = st.selectbox(
        "Select a movie:",
        options=movie_titles,
        index=0,
        key="movie_select"
    )
    
    # Number of recommendations
    num_recs = st.slider(
        "Number of recommendations:",
        min_value=1,
        max_value=10,
        value=5,
        key="num_slider"
    )
    
    # Genre filter
    all_genres = set()
    for genres in df['genres']:
        all_genres.update([g.strip() for g in genres.split(',')])
    genre_options = ["All"] + sorted(list(all_genres))
    
    selected_genre = st.selectbox(
        "Filter by genre (optional):",
        options=genre_options,
        key="genre_select"
    )
    
    # Recommendation button
    if st.button("Get Recommendations", type="primary"):
        st.session_state.get_recs = True
        st.session_state.selected_movie = selected_movie
        st.session_state.num_recs = num_recs
        st.session_state.selected_genre = selected_genre

with col2:
    if st.session_state.get_recs:
        st.markdown(f'<h3 class="sub-header">🎭 Movies similar to "{st.session_state.selected_movie}"</h3>', unsafe_allow_html=True)
        
        with st.spinner("Finding similar movies..."):
            recommendations = recommend_movies(
                st.session_state.selected_movie,
                st.session_state.num_recs,
                st.session_state.selected_genre
            )
        
        if not recommendations.empty:
            for idx, row in recommendations.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div class="movie-card">
                        <h4>🎬 {row['title']}</h4>
                        <p><strong>Genres:</strong> {row['genres']}</p>
                        <p><strong>Similarity Score:</strong> {row['similarity_score']:.3f}</p>
                        <p><strong>Overview:</strong> {row['overview']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No recommendations found. Try adjusting your filters.")

# Additional features
st.markdown("---")
st.markdown('<h3 class="sub-header">📊 Movie Database Overview</h3>', unsafe_allow_html=True)

col3, col4, col5 = st.columns(3)

with col3:
    st.metric("Total Movies", len(df))

with col4:
    unique_genres = len(all_genres)
    st.metric("Unique Genres", unique_genres)

with col5:
    avg_title_length = df['title'].str.len().mean()
    st.metric("Avg Title Length", f"{avg_title_length:.1f} chars")

# Display sample of the dataset
if st.checkbox("Show Movie Dataset"):
    st.dataframe(df[['title', 'genres']], use_container_width=True)

# Genre distribution
if st.checkbox("Show Genre Distribution"):
    genre_counts = {}
    for genres in df['genres']:
        for genre in genres.split(','):
            genre = genre.strip()
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    genre_df = pd.DataFrame(list(genre_counts.items()), columns=['Genre', 'Count'])
    genre_df = genre_df.sort_values('Count', ascending=False)
    
    st.bar_chart(genre_df.set_index('Genre'))

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🎬 Movie Recommendation System | Built with Streamlit & Scikit-learn</p>
    </div>
    """,
    unsafe_allow_html=True
)