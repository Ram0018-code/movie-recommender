import streamlit as st
import pandas as pd
import ast
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Page Config ---
st.set_page_config(
    page_title="Movie Genius",
    page_icon="🎬",
    layout="wide"
)

# --- 1. Load and Cache Data ---
@st.cache_data
def load_data():
    movies = pd.read_csv('tmdb_5000_movies.csv')

    # Try reading the zip file safely
    try:
        # 1. Try reading it directly
        credits = pd.read_csv('tmdb_5000_credits.zip', compression='zip')
    except:
        # 2. Fallback for Mac/Windows zip structure issues
        import zipfile
        with zipfile.ZipFile('tmdb_5000_credits.zip', 'r') as z:
            # Find the file that ends with .csv inside the zip
            csv_file = [f for f in z.namelist() if f.endswith('.csv') and '__MACOSX' not in f][0]
            credits = pd.read_csv(z.open(csv_file))

    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies.dropna(inplace=True)
    return movies

# --- 2. Process Data & Train Model ---
@st.cache_resource
def train_model(movies):
    def convert(obj):
        L = []
        try:
            for i in ast.literal_eval(obj):
                L.append(i['name'])
        except: pass
        return L

    def convert3(obj):
        L = []
        counter = 0
        try:
            for i in ast.literal_eval(obj):
                if counter != 3:
                    L.append(i['name'])
                    counter += 1
                else: break
        except: pass
        return L

    def fetch_director(obj):
        L = []
        try:
            for i in ast.literal_eval(obj):
                if i['job'] == 'Director':
                    L.append(i['name'])
                    break
        except: pass
        return L

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    for col in ['genres', 'keywords', 'cast', 'crew']:
        movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])

    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    
    new_df = movies[['movie_id', 'title', 'tags']]
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

    cv = CountVectorizer(max_features=2000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    similarity = cosine_similarity(vectors)
    
    return new_df, similarity

# --- 3. Function to Fetch Posters ---
def fetch_poster(movie_id):
    # ------------------------------------------------------------------
    # IMPORTANT: You need an API Key for real posters!
    # 1. Go to https://www.themoviedb.org/ -> Login -> Settings -> API
    # 2. Copy your API Key and paste it below inside the quotes.
    # ------------------------------------------------------------------
    api_key = "1ee38f5d72fa5c35f06e6b5b5ba5b364" 
    # Note: I've provided a sample key, but it's best to use your own!
    
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
        data = requests.get(url)
        data = data.json()
        poster_path = data['poster_path']
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
        return full_path
    except:
        # Returns a grey placeholder image if the API fails or key is invalid
        return "https://via.placeholder.com/500x750?text=No+Image+Available"

# --- Load everything ---
st.header("🎬 Movie Recommendation System")

with st.spinner('Loading Movie Database...'):
    raw_movies = load_data()
    df, similarity = train_model(raw_movies)

# --- Website UI ---
st.markdown("""
<style>
.big-font { font-size:20px !important; }
img { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.write("Select a movie you like, and we will suggest 5 others with similar plots, genres, and cast.")

selected_movie = st.selectbox(
    "Type or select a movie:",
    df['title'].values
)

if st.button('Show Recommendations'):
    try:
        movie_index = df[df['title'] == selected_movie].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        st.subheader(f"Because you watched {selected_movie}:")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        cols = [col1, col2, col3, col4, col5]
        
        # Loop through results
        for idx, col in enumerate(cols):
            movie_idx = movies_list[idx][0]
            movie_title = df.iloc[movie_idx].title
            movie_id = df.iloc[movie_idx].movie_id
            
            # Fetch the poster
            poster_url = fetch_poster(movie_id)
            
            with col:
                st.text(movie_title)
                st.image(poster_url)
                
    except Exception as e:
        st.error(f"Error: {e}")

st.sidebar.markdown("### About")
st.sidebar.info("This project uses Content-Based Filtering. It analyzes movie tags (plot, cast, director) to find mathematical similarities.")
