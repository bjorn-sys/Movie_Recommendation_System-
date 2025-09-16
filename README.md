🎬 Movie Recommendation Engine using Cosine Similarity
---
📌 Project Overview

This project builds a content-based recommendation system for movies using TF-IDF vectorization and cosine similarity.
The system suggests similar movies based on their genre and one-line descriptions.

It can:

Recommend movies similar to a given title

Filter recommendations by genre

Handle missing values, duplicates, and noisy text

Provide similarity scores for each recommendation
---
⚙️ Technologies Used

Python 3.x

Libraries: Pandas, NumPy, Scikit-learn, Regular Expressions (re)
---
📂 Dataset

The dataset used in this project contains 9,999 movie entries with the following key columns:

MOVIES → Title of the movie

GENRE → Genre(s) of the movie

ONE-LINE → Short plot description

After cleaning, the dataset has:

9,072 movies

510 unique genres  
---
🛠️ Project Workflow
1️⃣ Data Loading & Cleaning

Load the dataset

Drop missing values and duplicates

Clean text: lowercase conversion, special character removal, whitespace handling

---
2️⃣ Feature Engineering

Combine genre and one-line description into a new column content

Apply TF-IDF Vectorization with parameters to optimize vocabulary size and context

---

3️⃣ Similarity Calculation

Compute cosine similarity across all movies

Store results in a similarity matrix

---

4️⃣ Recommendation Engine

Input: Movie title

Output: Top N recommended movies with similarity scores

Optional: Filter recommendations by genre

---

🎯 Example Output

Input Movie: Blood Red Sky
Genre Filter: Horror

MOVIES	GENRE	Similarity Score
Flight	Drama, Horror, Sci-Fi	0.276
The Bad Batch	Action, Horror, Mystery	0.180

---
📌 Key Features

✅ Handles missing values and duplicates
✅ Cleans noisy text and standardizes format
✅ Uses TF-IDF + Cosine Similarity for recommendations
✅ Genre-based filtering available
✅ Easily extendable to other datasets
