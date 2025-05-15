import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load data
df = pd.read_csv('preprocessed_recipes.csv')

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['ingredients_text'])

# Function to suggest recipes
def suggest_recipes(user_input, top_n=3):
    user_input = user_input.lower()
    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, tfidf_matrix)
    top_indices = similarity[0].argsort()[::-1][:top_n]
    return df.iloc[top_indices][['recipe_name', 'ingredients_cleaned', 'directions_cleaned', 'rating']]

# Streamlit UI
st.title("🍽️ InstaCook - Ingredient Based Recipe Recommender")
st.write("Enter ingredients you have and get recipe suggestions instantly!")

user_input = st.text_input("Enter ingredients (comma separated):", "")

if st.button("Suggest Recipes"):
    if user_input.strip() == "":
        st.warning("Please enter some ingredients.")
    else:
        results = suggest_recipes(user_input)
        for idx, row in results.iterrows():
            st.subheader(row['recipe_name'])
            st.markdown(f"**Ingredients:** {', '.join(eval(row['ingredients_cleaned']))}")
            st.markdown(f"**Directions:** {row['directions_cleaned'][:300]}...")  # Show first 300 chars
            st.markdown(f"**Rating:** {row['rating']}/5")
            st.markdown("---")
