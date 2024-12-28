import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from nlp_utils import preprocess_text


movies_df = pd.read_csv("movie_recommendations_2024_final.csv")
movies_df.info()
movies_df.isnull().sum()


# Data Preprocessing and Analysis:
# Text Cleaning (NLP):
# Apply preprocessing to the 'Storyline' column
movies_df['cleaned_storyline'] = movies_df['StoryLine'].apply(preprocess_text)

movies_df['MovieName'] = movies_df['MovieName'].astype(str)
movies_df['StoryLine'] = movies_df['StoryLine'].astype(str)
movies_df['cleaned_storyline'] = movies_df['cleaned_storyline'].astype(str)


# Text Representation:
# In Natural Language Processing (NLP),
# text data needs to be converted into a numerical format because most machine learning models
# and algorithms can only work with numbers. This process is known as text representation.

# Count Vectorizer: Converts each document (in this case, movie storyline) into a vector of word counts.
# It creates a matrix where each row represents a document,
# and each column represents a word. The value in the matrix is the frequency of the word in the document.

# Initialize the Count Vectorizer
count_vectorizer = CountVectorizer(max_features=1000)

# Fit and transform the cleaned storylines into a Count Vectorizer matrix
count_matrix = count_vectorizer.fit_transform(movies_df['cleaned_storyline'])

# Convert the matrix into a DataFrame for easier inspection
count_df = pd.DataFrame(count_matrix.toarray(), columns=count_vectorizer.get_feature_names_out())

# Add the Count Vectorizer DataFrame as a new column in the original movies_df
movies_df['Count_Vector'] = list(count_df.values)

movies_df.head()

# Cosine Similarity:

from sklearn.metrics.pairwise import cosine_similarity
# Extract the Count_Vector column as a matrix
count_matrix = np.array(movies_df['Count_Vector'].to_list())

# Calculate the cosine similarity matrix
cosine_sim_matrix = cosine_similarity(count_matrix)


# Function to get ranked movie names for a given movie
def get_ranked_movies(index, cosine_sim_matrix, movies_df, top_n=5):
    # Get the similarity scores for the given movie
    sim_scores = list(enumerate(cosine_sim_matrix[index]))

    # Sort the movies by similarity score in descending order
    sorted_sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top N most similar movies (excluding itself)
    top_movies = sorted_sim_scores[1:top_n + 1]  # Skip the first one, as it's the movie itself

    # Retrieve the movie names
    ranked_movies = [movies_df.iloc[i]['MovieName'] for i, score in top_movies]
    return ranked_movies


# Add a new column with ranked movies
movies_df['Ranked_Movies'] = [
    get_ranked_movies(i, cosine_sim_matrix, movies_df) for i in range(len(movies_df))
]


movies_df = movies_df.drop(columns  = ['Unnamed: 0'])

movies_df['MovieName'] = movies_df['MovieName'].astype(str)
movies_df['StoryLine'] = movies_df['StoryLine'].astype(str)
movies_df['cleaned_storyline'] = movies_df['cleaned_storyline'].astype(str)

movies_df.to_csv("imdb_movies_data_final_003.csv")