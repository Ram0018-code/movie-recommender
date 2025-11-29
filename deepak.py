import streamlit as st
import pandas as pd
import requests
import pickle
import os
import zipfile
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# --- 1. CONFIGURATION (Must be first) ---
st.set_page_config(
    page_title="Movies Recommendation",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. CUSTOM CSS (Design & Animations) ---
st.markdown("""
<style>
    /* Dark Netflix-like Background */
    .stApp {
        background-color: #0e1117;
        background-image: linear-gradient(to right, #000000, #1a1a1a);
        color: #ffffff;
    }
    
    /* Red Title */
    h1 {
        font-family: 'Helvetica Neue', sans-serif;
        color: #e50914 !important; 
        text-shadow: 2px 2px 4px #000000;
        font-weight: 800;
        text-align: center;
    }
    
    /* Hide the default button style if any remain */
    .stButton > button {
        display: none;
    }
    
    /* Modern Search Box Styling */
    .stSelectbox > div[data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div[data-baseweb="select"] > div:hover {
        border-color: #e50914;
        background-color: rgba(255, 255, 255, 0.15);
        box-shadow: 0 0 10px rgba(229, 9, 20, 0.3);
    }
    
    .stSelectbox label {
        color: #b3b3b3 !important;
        font-size: 1rem;
        font-weight: 500;
    }
    
    /* Dropdown Menu Items */
    div[role="listbox"] ul {
        background-color: #141414;
        color: white;
    }
    
    /* Poster Hover Zoom with Smooth Animation */
    div[data-testid="stImage"] img {
        border-radius: 8px;
        transition: transform 0.4s ease-in-out, box-shadow 0.4s ease-in-out; /* Smooth transition */
    }
    div[data-testid="stImage"] img:hover {
        transform: scale(1.08); /* Slightly larger scale */
        cursor: pointer;
        box-shadow: 0 10px 20px rgba(0,0,0,0.5); /* Add shadow on hover */
        z-index: 10; /* Ensure it pops out above other elements */
    }
    
    /* Movie Title */
    .movie-title {
        font-weight: bold;
        text-align: center;
        margin-top: 8px;
        font-size: 1rem;
        color: white;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    /* Watch Provider Text */
    .provider-text {
        font-size: 0.75rem; 
        color: #b3b3b3; 
        text-align: center;
        margin-bottom: 5px;
        height: 30px; /* Fixed height to align buttons */
    }
    
    /* "Watch Now" Button Link */
    .watch-btn {
        display: block;
        margin: 0 auto;
        width: fit-content;
        color: #e50914; 
        border: 1px solid #e50914; 
        padding: 5px 15px; 
        border-radius: 4px; 
        text-decoration: none; 
        font-size: 0.8rem;
        font-weight: bold;
        transition: 0.3s;
    }
    .watch-btn:hover {
        background-color: #e50914;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. DATA FUNCTIONS ---

@st.cache_data
def load_data():
    # 1. Load Movies
    if not os.path.exists('tmdb_5000_movies.csv'):
        st.error("❌ Error: 'tmdb_5000_movies.csv' not found.")
        return None
    movies = pd.read_csv('tmdb_5000_movies.csv')
    
    # 2. Load Credits (Smart Universal Check)
    credits = None
    
    # Check A: Do we have the unzipped CSV? (Likely true locally)
    if os.path.exists('tmdb_5000_credits.csv'):
        credits = pd.read_csv('tmdb_5000_credits.csv')
        
    # Check B: Do we have the CSV.ZIP? (Likely true on GitHub)
    elif os.path.exists('tmdb_5000_credits.csv.zip'):
        with zipfile.ZipFile('tmdb_5000_credits.csv.zip', 'r') as z:
            csv_file = [f for f in z.namelist() if f.endswith('.csv') and '__MACOSX' not in f][0]
            credits = pd.read_csv(z.open(csv_file))
            
    # Check C: Do we have just .ZIP? (Alternate name)
    elif os.path.exists('tmdb_5000_credits.zip'):
        with zipfile.ZipFile('tmdb_5000_credits.zip', 'r') as z:
            csv_file = [f for f in z.namelist() if f.endswith('.csv') and '__MACOSX' not in f][0]
            credits = pd.read_csv(z.open(csv_file))
            
    if credits is None:
        st.error("❌ Error: Could not find 'tmdb_5000_credits' file.")
        return None

    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies.dropna(inplace=True)
    return movies

@st.cache_resource
def train_model(movies):
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

    # Limit to 2000 for Streamlit Cloud Memory Safety
    cv = CountVectorizer(max_features=2000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()
    similarity = cosine_similarity(vectors)
    
    return movies, similarity

# --- 4. API FUNCTIONS ---

def fetch_poster(movie_id):
    api_key = "8265bd1679663a7ea12ac168da84d2e8" # Use your own key if you have one
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
        data = requests.get(url).json()
        return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
    except:
        return "https://via.placeholder.com/500x750?text=No+Image"

def fetch_watch_providers(movie_id):
    api_key = "8265bd1679663a7ea12ac168da84d2e8"
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers?api_key={api_key}"
        data = requests.get(url).json()
        results = data.get('results', {})
        
        # Priority: India (IN) first, then US
        target_country = 'IN' if 'IN' in results else 'US'
        
        if target_country in results:
            provider_data = results[target_country]
            link = provider_data.get('link', '')
            
            # Get Streaming Services (Flatrate)
            providers = []
            if 'flatrate' in provider_data:
                providers = [p['provider_name'] for p in provider_data['flatrate']]
            
            # Return top 2 providers (e.g. ['Netflix', 'Amazon Prime']) and the link
            return providers[:2], link
            
        return None, None
    except:
        return None, None

# --- 5. MAIN APP UI ---

movies_raw = load_data()

if movies_raw is not None:
    df, similarity = train_model(movies_raw)

    st.title("Movies Recommendation")
    st.markdown("<div style='text-align: center; color: #b3b3b3; margin-bottom: 30px;'>Discover your next favorite movie</div>", unsafe_allow_html=True)

    # Search Box
    selected_movie = st.selectbox("Select a movie you like:", df['title'].values)

    st.write("---")

    # Automatic Recommendation Trigger
    if selected_movie:
        try:
            with st.spinner('Finding matches...'):
                movie_index = df[df['title'] == selected_movie].index[0]
                distances = similarity[movie_index]
                movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
                
                st.subheader(f"Because you watched {selected_movie}:")
                st.write("") 

                # 5 Columns for Recommendations
                cols = st.columns(5)
                
                for idx, col in enumerate(cols):
                    movie_idx = movies_list[idx][0]
                    movie_title = df.iloc[movie_idx].title
                    movie_id = df.iloc[movie_idx].movie_id
                    
                    poster = fetch_poster(movie_id)
                    providers, watch_link = fetch_watch_providers(movie_id)
                    
                    with col:
                        # 1. Poster Image
                        st.image(poster, use_container_width=True)
                        
                        # 2. Movie Title
                        st.markdown(f"<div class='movie-title' title='{movie_title}'>{movie_title}</div>", unsafe_allow_html=True)
                        
                        # 3. Streaming Info
                        if providers:
                            p_str = ", ".join(providers)
                            st.markdown(f"<div class='provider-text'>On: <b style='color:#46d369'>{p_str}</b></div>", unsafe_allow_html=True)
                            st.markdown(f"<a href='{watch_link}' target='_blank' class='watch-btn'>▶ Watch</a>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='provider-text'>Not streaming</div>", unsafe_allow_html=True)
                            
        except Exception as e:
            st.error(f"Error: {e}")

    # Footer
    st.write("")
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #666;'>Created by Deepak • Powered by TMDB</div>", unsafe_allow_html=True)

else:
    st.stop()
