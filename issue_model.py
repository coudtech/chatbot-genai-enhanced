import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from tqdm import tqdm
import cloudpickle as pickle  # Use cloudpickle instead of pickle

class IssueModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.kmeans = None
        self.tfidf_matrix = None
        self.df = None

    def train(self, filepath='data/mybook2.csv'):
        print("Loading data...")
        self.df = pd.read_csv(filepath, encoding='latin1')
        self.df.dropna(subset=['Description'], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.df['Index'] = self.df.index

        print("Vectorizing text...")
        descriptions = list(tqdm(self.df['Description'], desc="TF-IDF Vectorizing"))
        self.tfidf_matrix = self.vectorizer.fit_transform(descriptions)

        if 'App' in self.df.columns:
            print("Training KMeans...")
            app_counts = self.df['App'].value_counts()
            n_clusters = min(10, len(app_counts))
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            self.kmeans.fit(self.tfidf_matrix)

    def save(self, model_path='search_model.pkl'):
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)  # Saving with cloudpickle
        print(f"Model saved at {model_path}")
