import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Streamlit app title
st.title("Book Recommendation System")

# File upload
uploaded_file = st.file_uploader("book_data.csv", type=["csv"])
if uploaded_file:
    # Read the uploaded file
    data = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.write(data.head())

    # Keep only relevant columns
    if {"book_title", "book_desc", "book_rating_count"}.issubset(data.columns):
        data = data[["book_title", "book_desc", "book_rating_count"]]
        
        # Handle missing values
        st.write("Checking for missing values...")
        st.write(data.isnull().sum())
        data = data.dropna()

        # Sort by rating count and display the top 5
        data = data.sort_values(by="book_rating_count", ascending=False)
        top_5 = data.head()
        st.subheader("Top 5 Rated Books")
        st.write(top_5)

        # Plot Pie Chart for Top 5 Rated Books
        labels = top_5["book_title"]
        values = top_5["book_rating_count"]
        colors = ['gold', 'lightgreen']

        fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
        fig.update_layout(title_text="Top 5 Rated Books")
        fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=15,
                          marker=dict(colors=colors, line=dict(color='black', width=2)))
        st.plotly_chart(fig)

        # TF-IDF and Similarity Calculation
        st.subheader("Book Similarity Analysis")
        feature = data["book_desc"].tolist()
        tfidf = TfidfVectorizer(input="content", stop_words="english")
        tfidf_matrix = tfidf.fit_transform(feature)
        similarity = linear_kernel(tfidf_matrix, tfidf_matrix)

        # Interactive similarity finder
        selected_book = st.selectbox("Select a book to find similar ones:", data["book_title"])
        if selected_book:
            book_index = data[data["book_title"] == selected_book].index[0]
            sim_scores = list(enumerate(similarity[book_index]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            top_similar_books = [data.iloc[i[0]]["book_title"] for i in sim_scores[1:6]]
            
            st.write(f"Books similar to '{selected_book}':")
            for book in top_similar_books:
                st.write(f"- {book}")
    else:
        st.error("The uploaded file must contain 'book_title', 'book_desc', and 'book_rating_count' columns.")
else:
    st.info("Please upload a CSV file to begin.")
