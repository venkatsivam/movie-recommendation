from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt


# Function to recommend movies based on user input and precomputed similarity rankings
def compute_similarity(input_storyline, movies_df):
    # Combine input storyline with existing storylines
    storylines = movies_df['cleaned_storyline'].tolist()
    storylines.insert(0, input_storyline)  # Add user input at the beginning

    # Compute TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(storylines)

    # Compute cosine similarity
    similarity_scores = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])[0]

    # Add similarity scores to the dataframe
    movies_df['Similarity_Score'] = similarity_scores
    top_movies = movies_df.sort_values(by='Similarity_Score', ascending=False).head(5)
    top_similarity_scores = top_movies['Similarity_Score'].tolist()
    top_movies_list = top_movies['MovieName'].tolist()
    return top_movies, top_movies_list, top_similarity_scores


# Bar plot for similarity scores
def plot_similarity(similarity_scores, movie_names):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=similarity_scores, y=movie_names, palette="viridis")
    plt.title("Similarity Scores for Recommended Movies")
    plt.xlabel("Similarity Score")
    plt.ylabel("Movie Name")
    st.pyplot(plt)


