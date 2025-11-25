🍿 Content-Based Movie Recommendation System

A machine learning minor project deployed as a live web application using Streamlit. This system recommends movies based on semantic similarity of plot, genre, cast, and director.

🔗 Live Demo & Repository

Resource

Link

Live Web App

[https://deepak-movies.streamlit.app/]

Source Code

[https://lnkd.in/gUYeFxNv]

🎯 Model Overview (Content-Based Filtering)

This project uses Content-Based Filtering, which is a technique that recommends items (movies) based on a comparison of what the user likes with the features of the items themselves.

Architecture

Data Acquisition: Using the TMDB 5000 Movies and Credits datasets.

Preprocessing: Merging and extracting key features from JSON strings: genres, keywords, cast (top 3), and crew (director).

Feature Engineering: Creating a single 'Tags' column for each movie by concatenating the overview, genres, keywords, cast, and director. Spaces in names (e.g., Chris Evans) are removed to treat them as single entities (ChrisEvans).

Vectorization: The tags are converted into numerical vectors using CountVectorizer (Bag of Words technique, limited to 2000 features for memory efficiency).

Similarity Calculation: The distance between every movie vector is calculated using Cosine Similarity. A score of 1.0 means the movies are identical.

Recommendation: Given a movie, the system finds the 5 movies with the highest similarity scores.

⚙️ Technical Stack

Language: Python

Modeling: scikit-learn (for CountVectorizer and cosine_similarity)

Data Handling: Pandas and Numpy

Web Framework: Streamlit (for fast deployment and UI)

External Integration: requests library for fetching live movie posters via the TMDB API.

💻 Setup and Installation (Local Run)

To run this project on your local machine:

1. Clone the Repository

git clone [https://lnkd.in/gUYeFxNv]
cd movie-recommender


2. Prepare Data (Crucial Step)

Ensure the following files are in the root directory:

tmdb_5000_movies.csv

tmdb_5000_credits.csv (or the compressed file tmdb_5000_credits.csv.zip used for deployment)

3. Install Dependencies

Install all required libraries using the provided requirements.txt file:

pip install -r requirements.txt


4. Run the Application

Execute the Streamlit app. Note that the file name is deepak.py.

streamlit run deepak.py


The application will open in your default browser.

📝 Future Scope

Implement a Collaborative Filtering model (e.g., Matrix Factorization) to compare with Content-Based results.

Add movie ratings and user reviews to the displayed output.

Integrate user session management for saving personalized watchlists.

🤝 Contribution

Feel free to open issues or submit pull requests to improve this project!
