import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def _init_(self, file_path):
        self.file_path = file_path
        self.df = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.load_data()
        self.prepare_features()
        self.calculate_similarity()
    
    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        self.df = self.df[self.df['type'] == 'Movie'].reset_index(drop=True)
        for col in ['cast', 'director', 'listed_in', 'description']:
            self.df[col] = self.df[col].fillna('')
    
    def prepare_features(self):
        self.df['combined_features'] = (
            self.df['cast'] + ' ' +
            self.df['director'] + ' ' +
            self.df['listed_in'] + ' ' +
            self.df['description']
        )
        self.df['title_lower'] = self.df['title'].str.lower()
    
    def calculate_similarity(self):
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.df['combined_features'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
    
    def recommend(self, title, top_n=5):
        title = title.lower()
        if title not in self.df['title_lower'].values:
            return pd.DataFrame()  # biar ga error kalau film ga ada

        idx = self.df[self.df['title_lower'] == title].index[0]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        indices = [i[0] for i in sim_scores]
        return self.df[['title', 'listed_in', 'description']].iloc[indices].reset_index(drop=True)