import pickle
import streamlit as st
import requests
import pandas as pd
import numpy as np

cred = pd.read_csv('data/tmdb_5000_credits.csv')
mov = pd.read_csv('data/tmdb_5000_movies.csv')

def popularity_based(pop_id, n_rec):
    names = pop_id.original_title.to_numpy()
    ids = pop_id.id.to_numpy()
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in range(0, n_rec):
        recommended_movie_names.append(names[i])
        recommended_movie_posters.append(fetch_poster(ids[i]))
    return recommended_movie_names, recommended_movie_posters

def get_cf_ids(user, n_rec):
    ids = pd.DataFrame(mov['id']).to_numpy().tolist()
    predictions = []
    for i in range(0, 1):
        for j in range(0, len(ids)):
            predictions.append(svd_model.predict(uid=user, iid=ids[j][i], verbose=False))
    predictions.sort(key=lambda x: x.est, reverse=True)
    recommended_ids = []
    for i in range(0, n_rec):
        recommended_ids.append(predictions[i].iid)
    recommended_movie_names = []
    for i in recommended_ids:
        recommended_movie_names.append(movies.iloc[i].title)
    ids1 = [] 
    for i in recommended_movie_names:
        ids1.append(mov[mov['original_title'].isin([i])]['id'])
    ids1 = np.array(ids1).tolist()
    ids2 = []
    for i in ids1:
        for j in i:
            ids2.append(j)
    recommended_movie_posters = []
    for i in ids2:
        recommended_movie_posters.append(fetch_poster(i))
    return recommended_movie_names, recommended_movie_posters

def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=9a2cdc512c9de4ad83cc9c4c8abf9556".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

def recommend(movie, n_rec):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:n_rec+1]:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names,recommended_movie_posters

st.header('Movie Recommender System')
movies = pickle.load(open('model/movie_list.pkl','rb'))
similarity = pickle.load(open('model/similarity.pkl','rb'))
users_list = pickle.load(open('model/user-list.pkl', 'rb'))
svd_model = pickle.load(open('model/svd-model.pkl', 'rb'))
pop_id = pickle.load(open('model/pop-id.pkl', 'rb'))

rectype = ["Content Based", "Collaborative Based", "Trending", "Hybrid"]
selected_type = st.selectbox(
    "Select the recommendation type",
    rectype   
)

if selected_type=="Content Based" or selected_type=="Hybrid":
    movie_list = movies['title'].values
    selected_movie = st.selectbox(
        "Select a movie from the dropdown",
        movie_list
    )

elif selected_type=="Collaborative Based" or selected_type=="Hybrid":
    selected_user = st.selectbox(
        "Select your UserID",
        users_list
    )

else:
    pass
     
n_rec = st.slider('Slide to see more recommendations ', min_value=5, max_value=25)

if st.button('Show Recommendations'):
    if selected_type=="Content Based":
            cols = st.columns(n_rec)
            recommended_movie_names,recommended_movie_posters = recommend(selected_movie, n_rec)
            for i, (movie_name, movie_poster) in enumerate(zip(recommended_movie_names, recommended_movie_posters)):
                with cols[i]:
                    st.text(movie_name)
                    st.image(movie_poster)
    elif selected_type=="Collaborative Based":
            cols = st.columns(n_rec)
            recommended_movie_names,recommended_movie_posters = get_cf_ids(selected_user, n_rec)
            for i, (movie_name, movie_poster) in enumerate(zip(recommended_movie_names, recommended_movie_posters)):
                with cols[i]:
                    st.text(movie_name)
                    st.image(movie_poster)
    elif selected_type=="Trending":
            cols = st.columns(n_rec)
            recommended_movie_names,recommended_movie_posters = popularity_based(pop_id, n_rec)
            for i, (movie_name, movie_poster) in enumerate(zip(recommended_movie_names, recommended_movie_posters)):
                with cols[i]:
                    st.text(movie_name)
                    st.image(movie_poster)
    else:
            cols = st.columns(n_rec)
            recommended_movie_names,recommended_movie_posters = recommend(selected_movie, n_rec)
            recommended_movie_names1, recommended_movie_posters1 = get_cf_ids(selected_user, n_rec)
            recommended_movie_names2, recommended_movie_posters2 = popularity_based(pop_id, n_rec)
            for i, (movie_name, movie_poster) in enumerate(zip(recommended_movie_names, recommended_movie_posters)):
                with cols[i]:
                    st.text(movie_name)
                    st.image(movie_poster)

            for i, (movie_name, movie_poster) in enumerate(zip(recommended_movie_names1, recommended_movie_posters1)):
                with cols[i]:
                    st.text(movie_name)
                    st.image(movie_poster)

            for i, (movie_name, movie_poster) in enumerate(zip(recommended_movie_names2, recommended_movie_posters2)):
                with cols[i]:
                    st.text(movie_name)
                    st.image(movie_poster)