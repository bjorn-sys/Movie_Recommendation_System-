import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
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
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .recommendation-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .similarity-score {
        font-weight: bold;
        color: #2ca02c;
    }
    .stButton button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">🎬 Movie Recommendation System</h1>', unsafe_allow_html=True)
st.markdown("### Discover movies similar to your favorites using content-based filtering")

# Initialize session state for data persistence
if 'df' not in st.session_state:
    st.session_state.df = None
if 'indices' not in st.session_state:
    st.session_state.indices = None
if 'cosine_sim' not in st.session_state:
    st.session_state.cosine_sim = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Sidebar for navigation and information
with st.sidebar:
    st.header("About")
    st.info("""
    This recommendation system uses:
    - **TF-IDF Vectorization** to convert text into numerical features
    - **Cosine Similarity** to find movies with similar content
    - Content includes genre and plot descriptions
    """)
    
    st.header("Instructions")
    st.write("""
    1. Load your movie dataset (CSV format)
    2. Wait for the system to process the data
    3. Select a movie from the dropdown
    4. Get personalized recommendations!
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Movie Dataset (CSV)", type="csv")
    
    if uploaded_file is not None:
        with st.spinner("Loading and processing data..."):
            try:
                # Load the dataset
                df = pd.read_csv(uploaded_file)
                
                # Select relevant columns
                df = df[['MOVIES', 'GENRE', 'ONE-LINE']].copy()
                
                # Handle missing values
                df.dropna(inplace=True)
                
                # Remove duplicates
                df.drop_duplicates(inplace=True)
                
                # Text preprocessing function
                def preprocess_text(text):
                    if not isinstance(text, str):
                        return ""
                    text = text.lower()
                    text = re.sub(r'[^a-zA-Z\s]', '', text)
                    text = re.sub(r'\s+', ' ', text).strip()
                    return text
                
                # Apply preprocessing
                for col in ['MOVIES', 'GENRE', 'ONE-LINE']:
                    df[col] = df[col].apply(preprocess_text)
                
                # Remove entries with "add a plot"
                df = df[df['ONE-LINE'] != 'add a plot']
                
                # Create content feature
                df['content'] = df['GENRE'] + " " + df['ONE-LINE']
                
                # TF-IDF Vectorization
                vectorizer = TfidfVectorizer(
                    stop_words="english",
                    max_df=0.7,
                    min_df=2,
                    ngram_range=(1, 2),
                    max_features=10000
                )
                tfidf_matrix = vectorizer.fit_transform(df['content'])
                
                # Cosine similarity
                cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
                
                # Create indices mapping
                df = df.reset_index(drop=True)
                indices = pd.Series(df.index, index=df['MOVIES']).drop_duplicates()
                
                # Store in session state
                st.session_state.df = df
                st.session_state.indices = indices
                st.session_state.cosine_sim = cosine_sim
                st.session_state.data_loaded = True
                
                st.success("Data loaded and processed successfully!")
                
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
    else:
        st.info("Please upload a CSV file to get started")

# Main content area
if st.session_state.data_loaded:
    df = st.session_state.df
    indices = st.session_state.indices
    cosine_sim = st.session_state.cosine_sim
    
    # Recommendation function
    def recommend_movies(movie_title, n=5, genre_filter=None):
        if movie_title not in indices:
            similar_titles = [title for title in indices.index if movie_title.lower() in title.lower()]
            if similar_titles:
                return f"Movie '{movie_title}' not found. Did you mean: {', '.join(similar_titles[:3])}?"
            else:
                return f"Movie '{movie_title}' not found in dataset!"
        
        idx = indices[movie_title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n+6]
        
        movie_indices = [i[0] for i in sim_scores]
        similarity_values = [i[1] for i in sim_scores]
        
        recommendations = df[['MOVIES', 'GENRE']].iloc[movie_indices].copy()
        recommendations['Similarity_Score'] = similarity_values
        
        if genre_filter:
            genre_filter = genre_filter.lower()
            recommendations = recommendations[
                recommendations['GENRE'].str.lower().str.contains(genre_filter)
            ]
        
        return recommendations.head(n)
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<h3 class="sub-header">Find Similar Movies</h3>', unsafe_allow_html=True)
        
        # Movie selection dropdown
        movie_list = df['MOVIES'].tolist()
        selected_movie = st.selectbox("Select a movie:", movie_list)
        
        # Number of recommendations
        num_recommendations = st.slider("Number of recommendations:", 3, 10, 5)
        
        # Genre filter
        genre_options = ["All Genres"] + sorted(list(set(' '.join(df['GENRE'].unique()).split())))
        selected_genre = st.selectbox("Filter by genre (optional):", genre_options)
        
        # Recommendation button
        if st.button("Get Recommendations", use_container_width=True):
            st.session_state.get_recs = True
            st.session_state.selected_movie = selected_movie
            st.session_state.num_recs = num_recommendations
            st.session_state.selected_genre = selected_genre if selected_genre != "All Genres" else None
    
    with col2:
        if st.session_state.get_recs:
            st.markdown('<h3 class="sub-header">Recommended Movies</h3>', unsafe_allow_html=True)
            
            with st.spinner("Finding similar movies..."):
                if st.session_state.selected_genre:
                    recommendations = recommend_movies(
                        st.session_state.selected_movie, 
                        st.session_state.num_recs, 
                        st.session_state.selected_genre
                    )
                else:
                    recommendations = recommend_movies(
                        st.session_state.selected_movie, 
                        st.session_state.num_recs
                    )
            
            if isinstance(recommendations, str):
                st.warning(recommendations)
            else:
                for idx, row in recommendations.iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div class="recommendation-box">
                            <h4>{row['MOVIES'].title()}</h4>
                            <p><strong>Genre:</strong> {row['GENRE'].title()}</p>
                            <p><strong>Similarity Score:</strong> <span class="similarity-score">{row['Similarity_Score']:.3f}</span></p>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Display dataset statistics
    st.markdown("---")
    st.markdown('<h3 class="sub-header">Dataset Overview</h3>', unsafe_allow_html=True)
    
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    with stats_col1:
        st.metric("Total Movies", len(df))
    with stats_col2:
        st.metric("Unique Genres", df['GENRE'].nunique())
    with stats_col3:
        st.metric("Recommendation Model", "Content-Based")
    
    # Show sample of movies
    if st.expander("View Sample of Movies in Dataset"):
        st.dataframe(df[['MOVIES', 'GENRE']].head(10), use_container_width=True)

else:
    # Welcome message before data is loaded
    st.info("""
    👋 Welcome to the Movie Recommendation System!
    
    To get started:
    1. Upload your movie dataset CSV file using the sidebar
    2. Make sure your CSV contains columns named: MOVIES, GENRE, and ONE-LINE
    3. The system will process your data and prepare recommendations
    """)
    
    # Placeholder for demo purposes
    st.image("https://images.unsplash.com/photo-1489599102910-59206b8ca314?auto=format&fit=crop&q=80&w=2070", 
             caption="Movie Recommendation System", use_column_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Movie Recommendation System • Powered by TF-IDF and Cosine Similarity"
    "</div>", 
    unsafe_allow_html=True
)