import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import re
import pickle
import os

# download NLTK data if not already present
def download_nltk_data():
    """Downloads required NLTK packages"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

class TextPreprocessor:
    """Handles text cleaning and preprocessing"""
    
    def __init__(self, use_stemming=False, use_lemmatization=True, remove_stopwords=True):
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        
        if use_stemming:
            self.stemmer = PorterStemmer()
        
        if use_lemmatization:
            self.lemmatizer = WordNetLemmatizer()
        
        if remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Basic text cleaning - lowercase and remove special chars"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def tokenize_text(self, text):
        """Split text into words"""
        return word_tokenize(text)
    
    def process_tokens(self, tokens):
        """Apply stemming/lemmatization and remove stopwords"""
        processed_tokens = []
        
        for token in tokens:
            if self.remove_stopwords and token in self.stop_words:
                continue
            
            if self.use_stemming:
                token = self.stemmer.stem(token)
            
            if self.use_lemmatization:
                token = self.lemmatizer.lemmatize(token)
            
            processed_tokens.append(token)
        
        return processed_tokens
    
    def preprocess(self, text):
        """Full preprocessing pipeline"""
        text = self.clean_text(text)
        tokens = self.tokenize_text(text)
        tokens = self.process_tokens(tokens)
        return ' '.join(tokens)

class FeatureExtractor:
    """Extracts features using BOW or TF-IDF"""
    
    def __init__(self, method='tfidf', max_features=1000, ngram_range=(1, 1)):
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        
        if method == 'bow':
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range
            )
        elif method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def fit_transform(self, texts):
        """Fit and transform texts to feature vectors"""
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts):
        """Transform texts using fitted vectorizer"""
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        """Get list of features"""
        return self.vectorizer.get_feature_names_out()
    
    def save_vectorizer(self, filepath):
        """Save vectorizer for later use"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    def load_vectorizer(self, filepath):
        """Load saved vectorizer"""
        with open(filepath, 'rb') as f:
            self.vectorizer = pickle.load(f)


def prepare_data(df, test_size=0.2, random_state=42):
    """Split data into train/test sets with preprocessing"""
    print("\nPreparing data...")
    
    download_nltk_data()
    
    preprocessor = TextPreprocessor(
        use_stemming=False,
        use_lemmatization=True,
        remove_stopwords=True
    )
    
    print("Preprocessing texts...")
    df['processed_text'] = df['text'].apply(preprocessor.preprocess)
    
    # encode labels: Sport=0, Politics=1
    df['label'] = df['category'].map({'Sport': 0, 'Politics': 1})
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'],
        df['label'],
        test_size=test_size,
        random_state=random_state,
        stratify=df['label']
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, df

def extract_features_multiple_methods(X_train, X_test):
    """Extract features using BOW, TF-IDF with different n-grams"""
    features = {}
    
    print("\nExtracting features...")
    
    # bag of words
    print("1. Bag of Words...")
    bow_extractor = FeatureExtractor(method='bow', max_features=1000, ngram_range=(1, 1))
    X_train_bow = bow_extractor.fit_transform(X_train)
    X_test_bow = bow_extractor.transform(X_test)
    features['bow'] = {
        'X_train': X_train_bow,
        'X_test': X_test_bow,
        'extractor': bow_extractor,
        'name': 'Bag of Words (Unigrams)'
    }
    
    # TF-IDF unigrams
    print("2. TF-IDF...")
    tfidf_extractor = FeatureExtractor(method='tfidf', max_features=1000, ngram_range=(1, 1))
    X_train_tfidf = tfidf_extractor.fit_transform(X_train)
    X_test_tfidf = tfidf_extractor.transform(X_test)
    features['tfidf'] = {
        'X_train': X_train_tfidf,
        'X_test': X_test_tfidf,
        'extractor': tfidf_extractor,
        'name': 'TF-IDF (Unigrams)'
    }
    
    # TF-IDF with bigrams
    print("3. TF-IDF + Bigrams...")
    tfidf_bigram_extractor = FeatureExtractor(method='tfidf', max_features=1500, ngram_range=(1, 2))
    X_train_tfidf_bigram = tfidf_bigram_extractor.fit_transform(X_train)
    X_test_tfidf_bigram = tfidf_bigram_extractor.transform(X_test)
    features['tfidf_bigram'] = {
        'X_train': X_train_tfidf_bigram,
        'X_test': X_test_tfidf_bigram,
        'extractor': tfidf_bigram_extractor,
        'name': 'TF-IDF (Unigrams + Bigrams)'
    }
    
    # TF-IDF with trigrams
    print("4. TF-IDF + Trigrams...")
    tfidf_trigram_extractor = FeatureExtractor(method='tfidf', max_features=2000, ngram_range=(1, 3))
    X_train_tfidf_trigram = tfidf_trigram_extractor.fit_transform(X_train)
    X_test_tfidf_trigram = tfidf_trigram_extractor.transform(X_test)
    features['tfidf_trigram'] = {
        'X_train': X_train_tfidf_trigram,
        'X_test': X_test_tfidf_trigram,
        'extractor': tfidf_trigram_extractor,
        'name': 'TF-IDF (Unigrams + Bigrams + Trigrams)'
    }
    
    print("\nFeature extraction complete")
    return features

def save_features(features, y_train, y_test, output_dir='models/features'):
    """Save features and labels to disk"""
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    for key, feature_dict in features.items():
        vectorizer_path = os.path.join(output_dir, f'{key}_vectorizer.pkl')
        feature_dict['extractor'].save_vectorizer(vectorizer_path)
    
    print(f"\nSaved features to {output_dir}")

def main():
    """Run feature extraction pipeline"""
    print("=" * 60)
    print("Feature Extraction")
    print("=" * 60)
    
    df = pd.read_csv('data/dataset.csv')
    print(f"\nLoaded {len(df)} samples")
    
    X_train, X_test, y_train, y_test, df = prepare_data(df)
    features = extract_features_multiple_methods(X_train, X_test)
    save_features(features, y_train, y_test)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    
    return features, y_train, y_test

if __name__ == "__main__":
    features, y_train, y_test = main()
