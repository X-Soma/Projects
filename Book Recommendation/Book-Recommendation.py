import numpy as np
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import linear_kernel

data = pd.read_csv("book_data.csv")
print(data.head)
data = data[["book_title", "book_desc","book_rating_count"]]
print(data.head())
data = data.sort_values(by="book_rating_count",ascending=False)
top_5 = data.head()

import plotly.express as px
import plotly.graph_objects as go

labels = top_5["book_title"]
values = top_5["book_rating_count"]
colors = ['gold','lightgreen']

fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
fig.update_layout(title_text="Top 5 Rated Books")
fig.update_traces(hoverinfo='label+percent', textinfo ='percent',textfont_size=30,
                  marker=dict(colors=colors, line=dict(color='black',width=3)))
fig.show()
print(data.isnull().sum())
data = data.dropna()
feature = data["book_desc"].tolist()
tfidf = text.TfidfVectorizer(input=feature, stop_words="english")
tfidf_matrix = tfidf.fit_transform(feature)
similarity = linear_kernel(tfidf_matrix, tfidf_matrix)