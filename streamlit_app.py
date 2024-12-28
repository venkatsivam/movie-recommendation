import streamlit as st
import pandas as pd
from utils import compute_similarity, plot_similarity


movies_df = pd.read_csv("C:/Users/DELL/Desktop/AIML/accessments/imdb_movies_data_final_003.csv")  # data extracted from IMDB website using selenium web scrapper
print(movies_df.dtypes)

# When the user clicks the submit button
# Streamlit app layout
st.title("Movie Recommendation System")
st.write("Find the top 5 recommended movies based on your storyline input!")

# Input section
user_storyline = st.text_area("Enter a movie storyline:", placeholder="Write a short storyline or description...")
if st.button("Find Recommendations"):
    if user_storyline.strip():
        top_movies, top_movies_list, top_similarity_scores = compute_similarity(user_storyline, movies_df)

        # Display recommendations
        st.write("### Top 5 Recommended Movies:")
        for i, row in top_movies.iterrows():
            st.write(f"**{row['MovieName']}**")
            st.write(f"*Storyline*: {row['StoryLine']}")
            st.write("---")
        plot_similarity(top_similarity_scores, top_movies_list)
    else:
        st.error("Please enter a storyline to get recommendations!")
