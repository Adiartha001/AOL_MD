import streamlit as st
from movie_recommender import MovieRecommender

recommender = MovieRecommender("netflix_titles.csv")

st.title("🎬 Netflix Movie Recommender")
st.write("Masukkan judul film, dan sistem akan memberikan 5 rekomendasi mirip berdasarkan kontennya.")

title = st.text_input("Masukkan judul film:")

if st.button("Rekomendasikan"):
    result = recommender.recommend(title)
    
    if result.empty:
        st.warning("Film tidak ditemukan dalam dataset. Coba judul lain.")
    else:
        st.success("Berikut rekomendasi film:")
        for i, row in result.iterrows():
            st.subheader(row['title'])
            st.caption(f"Genre: {row['listed_in']}")
            st.write(row['description'])