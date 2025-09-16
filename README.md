# Movie_Recommendation_System-
Movie Recommendation System
A content-based movie recommendation system that suggests similar movies based on genre and plot descriptions using TF-IDF vectorization and cosine similarity.

📋 Table of Contents
Overview

Features

How It Works

Installation

Usage

Data Requirements

Algorithm Details

Results

Customization

Limitations

Future Improvements

🎬 Overview
This recommendation system uses content-based filtering to suggest movies similar to a user's selection. By analyzing movie genres and plot descriptions, the system identifies patterns and similarities between films to provide personalized recommendations.

✨ Features
Content-Based Filtering: Recommends movies based on similarity of content rather than user ratings

Genre Filtering: Option to filter recommendations by specific genres

Similarity Scoring: Shows how closely each recommendation matches the selected movie

Error Handling: Provides helpful suggestions when movie titles aren't found

Data Cleaning: Automated preprocessing of movie data

🔧 How It Works
The system follows these steps:

Data Loading: Reads movie data from a CSV file

Preprocessing: Cleans text data and handles missing values

Feature Engineering: Combines genre and plot descriptions into a single content feature

Vectorization: Converts text to numerical features using TF-IDF

Similarity Calculation: Computes cosine similarity between all movies

Recommendation Generation: Finds the most similar movies to the user's selection

📥 Installation
Prerequisites
Python 3.7+

Required libraries: pandas, numpy, scikit-learn

Setup
Clone or download the project files

Install required packages:

bash
pip install pandas numpy scikit-learn
Prepare your movie dataset in CSV format (see Data Requirements below)

🚀 Usage
Basic Implementation
python
# Import the recommendation function
from movie_recommender import recommend_movies

# Get recommendations for a movie
recommendations = recommend_movies("blood red sky", n=5)

# Get genre-specific recommendations
horror_recommendations = recommend_movies("blood red sky", n=5, genre_filter="horror")
Command Line Usage
Run the script directly:

bash
python movie_recommender.py
The system will process the data and provide recommendations for the test movie.

📊 Data Requirements
Your CSV file should include at least these columns:

MOVIES: Movie titles

GENRE: Genre information (comma-separated)

ONE-LINE: Brief plot descriptions

Example Data Structure
MOVIES	GENRE	ONE-LINE
Blood Red Sky	Action, Horror, Thriller	A woman with a mysterious illness is forced...
The Walking Dead	Drama, Horror, Thriller	Sheriff Deputy Rick Grimes wakes up from a...
🧠 Algorithm Details
TF-IDF Vectorization
Term Frequency-Inverse Document Frequency (TF-IDF) is used to convert text data into numerical vectors that represent the importance of words in the context of the entire dataset.

Parameters used:

stop_words="english": Removes common English words

max_df=0.7: Ignores terms that appear in more than 70% of documents

min_df=2: Ignores terms that appear in fewer than 2 documents

ngram_range=(1, 2): Considers both single words and word pairs

max_features=10000: Limits the number of features for efficiency

Cosine Similarity
Cosine similarity measures the cosine of the angle between two vectors in a multi-dimensional space. It's particularly effective for text similarity as it focuses on the direction rather than the magnitude of vectors.

Formula:

text
cosine_similarity(A, B) = (A · B) / (||A|| * ||B||)
📈 Results
The system successfully processes movie data and provides recommendations with similarity scores. For example:

Input: "blood red sky" (Action, Horror, Thriller)

Output:

text
MOVIES             GENRE                     SIMILARITY_SCORE
flight             drama horror scifi        0.276
the bad batch      action horror mystery     0.180
🛠 Customization
Adjusting Recommendation Parameters
You can modify the TF-IDF parameters in the code:

python
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,        # Adjust to control common word filtering
    min_df=2,          # Adjust to control rare word filtering
    ngram_range=(1, 2),# Change to (1, 1) for only single words
    max_features=10000 # Increase for larger datasets
)
Adding New Features
To incorporate additional movie features:

python
# Add additional features to the content
⚠️ Limitations
Cold Start Problem: Cannot recommend movies without sufficient content data

Content Dependency: Relies on accurate genre and plot information

Overspecialization: May recommend very similar movies without diversity

Text Quality: Performance depends on the quality of plot descriptions

🔮 Future Improvements
Potential enhancements for the system:

Hybrid Approach: Combine with collaborative filtering for better recommendations

Sentiment Analysis: Incorporate review sentiment into recommendations

Deep Learning: Use neural networks for more sophisticated feature extraction

User Interface: Develop a web or mobile app for easier interaction

Real-time Updates: Implement streaming data processing for new movies
Network URL: http://10.203.161.243:8501
