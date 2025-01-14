import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# --- Function to perform all the data processing and similarity tasks ---
def analyze_book_data(data):
    # Keep only relevant columns, handle errors
    required_cols = {"book_title", "book_desc", "book_rating_count"}
    if not required_cols.issubset(data.columns):
        st.error(f"The dataset must contain '{', '.join(required_cols)}' columns.")
        return None  # Return None to signal an error and prevent further processing

    data = data[list(required_cols)].copy()  # Copy to avoid SettingWithCopyWarning

    # Handle missing values
    st.write("Checking for missing values...")
    st.write(data.isnull().sum())
    data.dropna(inplace=True)

    # Sort by rating count and display the top 5
    data.sort_values(by="book_rating_count", ascending=False, inplace=True)
    top_5 = data.head().copy() # Copy to avoid SettingWithCopyWarning
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

    return data, similarity


# --- Streamlit App ---
st.title("Book Recommendation System")


# Default sample DataFrame
data = pd.DataFrame({
    'book_title': [
        'The Hitchhiker\'s Guide to the Galaxy',
        'Pride and Prejudice',
        'To Kill a Mockingbird',
        '1984',
        'The Lord of the Rings',
        'Jane Eyre',
         'The Great Gatsby',
         'Harry Potter and the Sorcerer\'s Stone',
         'The Da Vinci Code',
         'The Girl with the Dragon Tattoo'
    ],
    'book_desc': [
        'A humorous science fiction series about a man who travels the galaxy.',
        'A classic romantic novel about love, class, and society.',
        'A novel about childhood, prejudice, and justice in the American South.',
        'A dystopian novel about totalitarianism and government control.',
        'An epic fantasy series about the battle between good and evil.',
        'A gothic novel about love, secrets, and a strong heroine.',
        'A tragic love story set during the Jazz Age, exploring themes of wealth and the American Dream.',
        'The first book in a beloved series about a boy wizard.',
        'A mystery thriller about symbology and secret societies.',
        'A gripping crime thriller featuring a resourceful female hacker.'
    ],
    'book_rating_count': [500, 450, 480, 550, 600, 400, 475, 575, 525, 490]
})


st.subheader("Sample Data")
st.write(data.head())


analysis_results = analyze_book_data(data)

if analysis_results:
    data, similarity = analysis_results

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