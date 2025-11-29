import streamlit as st
import pandas as pd
import requests
import pickle
import os
import zipfile

# --- 1. CONFIGURATION (Must be first) ---
st.set_page_config(
    page_title="CineMatch AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. CUSTOM CSS (The Design) ---
# This injects CSS directly into the app to style buttons, background, and cards
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        background-image: linear-gradient(to right, #0f2027, #203a43, #2c5364); 
        color: #ffffff;
    }
    
    /* Title Styling */
    h1 {
        font-family: 'Helvetica Neue', sans-serif;
        color: #e50914 !important; /* Netflix Red */
        text-shadow: 2px 2px 4px #000000;
        font-weight: 800;
        font-size: 3.5rem !important;
        text-align: center;
        padding-bottom: 20px;
    }
    
    /* Dropdown Styling */
    .stSelectbox label {
        color: #ffffff !important;
        font-size: 1.2rem;
    }
    
    /* Button Styling */
    .stButton > button {
        background-color: #e50914; /* Netflix Red */
        color: white;
        border-radius: 5px;
        height: 3em;
        width: 100%;
        border: none;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #b20710;
        transform: scale(1.02);
        box-shadow: 0px 4px 15px rgba(229, 9, 20, 0.4);
    }
    
    /* Movie Card Effect */
    div[data-testid="stImage"] {
        border-radius: 10px;
        transition: transform 0.3s ease;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.5);
    }
    
    div[data-testid="stImage"]:hover {
        transform: scale(1.05);
        z-index: 1;
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.8);
    }
    
    /* Text Styling */
    p {
        font-size: 1.1rem;
    }
    
    .movie-title {
        font-weight: bold;
        text-align: center;
        margin-top: 10px;
        font-size: 1rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. FUNCTIONS ---

@st.cache_data
def load_data():
    # Universal Load Logic (Works locally and on cloud)
    if not os.path.exists('tmdb_5000_movies.csv'):
        st.error("❌ Error: 'tmdb_5000_movies.csv' not found.")
        return None
    
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = None
    
    # Check for various versions of the credits file
    if os.path.exists('tmdb_5000_credits.csv'):
        credits = pd.read_csv('tmdb_5000_credits.csv')
    elif os.path.exists('tmdb_5000_credits.csv.zip'):
        with zipfile.ZipFile('tmdb_5000_credits.csv.zip', 'r') as z:
            csv_file = [f for f in z.namelist() if f.endswith('.csv') and '__MACOSX' not in f][0]
            credits = pd.read_csv(z.open(csv_file))
    elif os.path.exists('tmdb_5000_credits.zip'):
        with zipfile.ZipFile('tmdb_5000_credits.zip', 'r') as z:
            csv_file = [f for f in z.namelist() if f.endswith('.csv') and '__MACOSX' not in f][0]
            credits = pd.read_csv(z.open(csv_file))
            
    if credits is None:
        st.error("❌ Error: Could not find credits file.")
        return None

    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies.dropna(inplace=True)
    return movies

@st.cache_resource
def train_model(movies):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import ast

    def convert(obj):
        try: return [i['name'] for i in ast.literal_eval(obj)]
        except: return []

    def convert3(obj):
        try: return [i['name'] for i in ast.literal_eval(obj)][:3]
        except: return []

    def fetch_director(obj):
        try: return [i['name'] for i in ast.literal_eval(obj) if i['job'] == 'Director'][:1]
        except: return []

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    movies['tags'] = movies['tags'].apply(lambda x: " ".join(x).lower())

    # Reduced features for speed/memory
    cv = CountVectorizer(max_features=2000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()
    similarity = cosine_similarity(vectors)
    
    return movies, similarity

def fetch_poster(movie_id):
    # USE YOUR OWN API KEY HERE
    api_key = "8265bd1679663a7ea12ac168da84d2e8"
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
        data = requests.get(url)
        data = data.json()
        poster_path = data['poster_path']
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    except:
        return "https://via.placeholder.com/500x750?text=No+Image"

# --- 4. APP LAYOUT ---

# Load Data
movies_raw = load_data()
if movies_raw is not None:
    df, similarity = train_model(movies_raw)

    st.title("🍿 CineMatch AI")
    st.markdown("<h4 style='text-align: center; color: #b3b3b3;'>Discover your next favorite movie with AI</h4>", unsafe_allow_html=True)
    st.write("") # Spacer

    # Search Bar Container
    with st.container():
        selected_movie = st.selectbox(
            "Select a movie from the library:",
            df['title'].values
        )

    st.write("") # Spacer
    
    # Recommendation Button
    if st.button('🚀 Recommend Movies'):
        with st.spinner('Analyzing plot, cast, and genres...'):
            try:
                movie_index = df[df['title'] == selected_movie].index[0]
                distances = similarity[movie_index]
                movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
                
                st.write("")
                st.subheader(f"Because you watched '{selected_movie}':")
                st.write("")

                # Dynamic Columns
                cols = st.columns(5)
                
                for idx, col in enumerate(cols):
                    movie_idx = movies_list[idx][0]
                    movie_title = df.iloc[movie_idx].title
                    movie_id = df.iloc[movie_idx].movie_id
                    poster = fetch_poster(movie_id)
                    
                    with col:
                        # Display Image
                        st.image(poster, use_container_width=True)
                        # Custom HTML for centered title
                        st.markdown(f"<div class='movie-title'>{movie_title}</div>", unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"Error: {e}")

    # Footer
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: grey;'>Created by Deepak | Powered by TMDB API & Streamlit</p>", unsafe_allow_html=True)

else:
    st.stop()
