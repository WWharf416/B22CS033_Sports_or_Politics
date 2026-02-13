import pickle
import numpy as np
from feature_extraction import TextPreprocessor

class TextClassifier:
    """
    Wrapper class for making predictions on new text
    """
    
    def __init__(self, model_path, vectorizer_path):
        """
        Initialize classifier with trained model and vectorizer
        
        Parameters:
        - model_path: path to saved model (.pkl file)
        - vectorizer_path: path to saved vectorizer (.pkl file)
        """
        self.preprocessor = TextPreprocessor(
            use_stemming=False,
            use_lemmatization=True,
            remove_stopwords=True
        )
        
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load vectorizer
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        self.labels = {0: 'Sport', 1: 'Politics'}
    
    def preprocess_text(self, text):
        """Preprocess input text"""
        return self.preprocessor.preprocess(text)
    
    def predict(self, text):
        """
        Predict category for input text
        
        Returns:
        - prediction: 'Sport' or 'Politics'
        - confidence: probability of predicted class
        """
        # Preprocess
        processed_text = self.preprocess_text(text)
        
        # Vectorize
        X = self.vectorizer.transform([processed_text])
        
        # Predict
        prediction = self.model.predict(X)[0]
        
        # Get probability if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)[0]
            confidence = probabilities[prediction]
        else:
            confidence = 1.0
        
        return self.labels[prediction], confidence
    
    def predict_batch(self, texts):
        """
        Predict categories for multiple texts
        
        Returns:
        - predictions: list of ('Sport' or 'Politics', confidence) tuples
        """
        results = []
        for text in texts:
            prediction, confidence = self.predict(text)
            results.append((prediction, confidence))
        
        return results

def load_best_model():
    """
    Load the best performing model based on training results
    """
    import pandas as pd
    
    # Read comparison results
    df_comparison = pd.read_csv('results/model_comparison.csv')
    
    # Find best model
    best_idx = df_comparison['Accuracy'].idxmax()
    best_model_info = df_comparison.loc[best_idx]
    
    feature_method = best_model_info['Feature Method']
    model_name = best_model_info['Model']
    
    print(f"Loading best model:")
    print(f"  Feature Method: {feature_method}")
    print(f"  Model: {model_name}")
    print(f"  Accuracy: {best_model_info['Accuracy']:.4f}")
    
    # Construct paths
    feature_key = feature_method.lower().replace(' ', '_')
    model_key = model_name.lower().replace(' ', '_')
    
    # Try to find the files
    model_dir = f"models/trained/{feature_key}"
    model_path = f"{model_dir}/{model_key}.pkl"
    
    # Determine vectorizer file based on feature method
    if 'bigram' in feature_method.lower() and 'trigram' in feature_method.lower():
        vectorizer_file = 'tfidf_trigram_vectorizer.pkl'
    elif 'bigram' in feature_method.lower():
        vectorizer_file = 'tfidf_bigram_vectorizer.pkl'
    elif 'tfidf' in feature_method.lower():
        vectorizer_file = 'tfidf_vectorizer.pkl'
    else:
        vectorizer_file = 'bow_vectorizer.pkl'
    
    vectorizer_path = f"models/features/{vectorizer_file}"
    
    return TextClassifier(model_path, vectorizer_path)

def predict_from_file(classifier, filepath):
    """
    Predict category for text from a file
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    prediction, confidence = classifier.predict(text)
    
    print(f"\nFile: {filepath}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.4f}")
    
    return prediction, confidence

def interactive_prediction(classifier):
    """
    Interactive mode for predicting text categories
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE PREDICTION MODE")
    print("=" * 60)
    print("\nEnter text to classify (or 'quit' to exit)")
    print("You can also enter 'file:<path>' to classify a text file")
    print("-" * 60)
    
    while True:
        print("\nEnter text:")
        user_input = input("> ").strip()
        
        if user_input.lower() == 'quit':
            print("Exiting...")
            break
        
        if user_input.lower().startswith('file:'):
            filepath = user_input[5:].strip()
            try:
                prediction, confidence = predict_from_file(classifier, filepath)
            except FileNotFoundError:
                print(f"Error: File '{filepath}' not found")
            except Exception as e:
                print(f"Error: {e}")
        
        elif user_input:
            prediction, confidence = classifier.predict(user_input)
            print(f"\nPrediction: {prediction}")
            print(f"Confidence: {confidence:.4f}")
        else:
            print("Please enter some text")

def demo_predictions(classifier):
    """
    Demonstrate predictions on sample texts
    """
    print("\n" + "=" * 60)
    print("DEMO PREDICTIONS")
    print("=" * 60)
    
    sample_texts = [
        "The basketball team won the championship game with a last-second three-pointer.",
        "The government announced new tax reforms to boost economic growth.",
        "Marathon runner sets new world record at Olympic Games.",
        "Parliament votes on healthcare legislation amid heated debate.",
        "Tennis star advances to finals after defeating top seed.",
        "President addresses nation on foreign policy and trade agreements."
    ]
    
    for i, text in enumerate(sample_texts, 1):
        prediction, confidence = classifier.predict(text)
        
        print(f"\n{i}. Text: {text[:70]}...")
        print(f"   Prediction: {prediction}")
        print(f"   Confidence: {confidence:.4f}")

def main():
    """
    Main prediction interface
    """
    print("=" * 60)
    print("SPORT vs POLITICS TEXT CLASSIFIER - PREDICTION")
    print("=" * 60)
    
    try:
        # Load best model
        classifier = load_best_model()
        
        # Run demo
        demo_predictions(classifier)
        
        # Interactive mode
        print("\n")
        choice = input("Would you like to try interactive mode? (y/n): ").strip().lower()
        
        if choice == 'y':
            interactive_prediction(classifier)
        
    except FileNotFoundError as e:
        print(f"\nError: Required files not found - {e}")
        print("Please run the main pipeline first: python main_pipeline.py")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
