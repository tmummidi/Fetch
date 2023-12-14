import subprocess
import sys

# List of packages to install or upgrade
packages_to_install = [
    'dash', 'dash-core-components', 'dash-html-components',
    'dash', 'pandas', 'scikit-learn', 'nltk', 'dash-bootstrap-components'
]

# Install or upgrade packages
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade"] + packages_to_install)
import dash
from dash import dcc, html, Input, Output
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
import difflib
import nltk
import dash_bootstrap_components as dbc

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set up stopwords and stemmer
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Function to calculate similarity score
def similarity_score(offer1, offer2):
    seq = difflib.SequenceMatcher(None, offer1, offer2)
    return seq.ratio()

# Function to remove duplicates with a higher score
def remove_duplicates_with_higher_score(recommendations):
    unique_offers = {}

    for offer, score in recommendations:
        duplicate_found = False
        for existing_offer, existing_score in unique_offers.items():
            if similarity_score(offer, existing_offer) > 0.8:
                if score > existing_score:
                    unique_offers[offer] = score
                    del unique_offers[existing_offer]
                duplicate_found = True
                break

        if not duplicate_found:
            unique_offers[offer] = score

    unique_recommendations = list(unique_offers.items())

    return unique_recommendations

# Function to load data from CSV files
def load_data(offer_path, brand_category_path, categories_path):
    df = pd.read_csv(offer_path)
    df1 = pd.read_csv(brand_category_path)
    df2 = pd.read_csv(categories_path)

    df1 = pd.merge(df1, df2[['PRODUCT_CATEGORY', 'IS_CHILD_CATEGORY_TO']], how='left',
                   left_on='BRAND_BELONGS_TO_CATEGORY', right_on='PRODUCT_CATEGORY')
    df = pd.merge(df, df1[['BRAND', 'BRAND_BELONGS_TO_CATEGORY', 'IS_CHILD_CATEGORY_TO']], on='BRAND', how='left')

    return df, df1, df2

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to preprocess text
def preprocess_text(text):
    words = word_tokenize(str(text))
    words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words]
    return ' '.join(words)

# Function to extract keywords
def extract_keywords(input_text):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([input_text])
    return vectorizer.get_feature_names_out()

# Function to recommend offers
def recommend_offers(input_text, tfidf_vectorizer, tfidf_matrix, df):
    input_text = preprocess_text(input_text)
    input_vector = tfidf_vectorizer.transform([input_text])

    cosine_similarities = linear_kernel(tfidf_matrix, input_vector).flatten()
    related_indices = cosine_similarities.argsort()[::-1]

    input_keywords = extract_keywords(input_text)

    unique_offers = set()
    recommendations = []

    for idx in related_indices:
        offer_keywords = extract_keywords(df['combined_text'].iloc[idx])
        brand_retailer_match = any(keyword in offer_keywords for keyword in input_keywords)

        if brand_retailer_match:
            offer_name = df['OFFER'].iloc[idx]
            similarity_score = cosine_similarities[idx]

            if offer_name not in unique_offers:
                recommendations.append((offer_name, similarity_score))
                unique_offers.add(offer_name)

    return recommendations

# Function to update output
def update_output(query):
    recommendations = recommend_offers(query, tfidf_vectorizer, tfidf_matrix, df)
    unique_recommendations = remove_duplicates_with_higher_score(recommendations)
    
    if unique_recommendations:
        return [html.P(f"Offer: {offer}, Similarity Score: {score}") for offer, score in unique_recommendations]
    else:
        return [html.P("No relevant offers found.")]

# Define app layout
app.layout = html.Div([
    html.H1("OFFER RECOMMENDATION DASHBOARD", style={'textAlign': 'center'}),

    dbc.Input(id='user-input', type='text', placeholder='Enter your search query', style={'width': '80%', 'margin': '10px'}),
    
    dbc.Button('Submit', id='submit-button', n_clicks=0, style={'margin': '10px'}),

    dcc.Loading(
        id="loading",
        type="circle",
        children=[html.Div(id='output-div', style={'margin': '20px', 'padding': '10px', 'border': '1px solid #ddd'})]
    )
])

# Callback to update output div
@app.callback(
    Output('output-div', 'children'),
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('user-input', 'value')]
)
def update_output_div(n_clicks, value):
    if n_clicks > 0:
        return update_output(value)

# Load data and run the app
if __name__ == '__main__':
    offer_path = 'offer_retailer.csv'
    brand_category_path = 'brand_category.csv'
    categories_path = 'categories.csv'

    try:
        df, df1, df2 = load_data(offer_path, brand_category_path, categories_path)
    except FileNotFoundError:
        print("Error: One or more files not found. Please check the file paths.")
        raise

    df['processed_offer'] = df['OFFER'].apply(preprocess_text)
    df['processed_retailer'] = df['RETAILER'].apply(preprocess_text)
    df['processed_brand'] = df['BRAND'].apply(preprocess_text)
    df['processed_BRAND_BELONGS_TO_CATEGORY'] = df['BRAND_BELONGS_TO_CATEGORY'].apply(preprocess_text)
    df['processed_IS_CHILD_CATEGORY_TO'] = df['IS_CHILD_CATEGORY_TO'].apply(preprocess_text)
    df['combined_text'] = df[['processed_offer', 'processed_retailer', 'processed_brand',
                              'processed_BRAND_BELONGS_TO_CATEGORY', 'processed_IS_CHILD_CATEGORY_TO']].agg(' '.join, axis=1)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_text'].astype(str))

    host = '127.0.0.1'
    port = 9030  # Change port if there is a connection issue

    print(f" * Running on http://{host}:{port}/ (Press CTRL+C to quit)")
    
    app.run_server(debug=True, port=9030)
