import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import pickle
import os
import time

class ModelTrainer:
    """Train and evaluate ML models"""
    
    def __init__(self, model_type, **kwargs):
        self.model_type = model_type
        self.model = self._initialize_model(**kwargs)
        self.training_time = None
        self.prediction_time = None
    
    def _initialize_model(self, **kwargs):
        """Initialize the specified model"""
        if self.model_type == 'naive_bayes':
            return MultinomialNB(**kwargs)
        
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(max_iter=1000, random_state=42, **kwargs)
        
        elif self.model_type == 'svm':
            return SVC(kernel='linear', random_state=42, probability=True, **kwargs)
        
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(random_state=42, **kwargs)
        
        elif self.model_type == 'knn':
            return KNeighborsClassifier(**kwargs)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train):
        """Train the model"""
        print(f"\nTraining {self.model_type}...")
        start_time = time.time()
        
        self.model.fit(X_train, y_train)
        
        self.training_time = time.time() - start_time
        print(f"Training time: {self.training_time:.2f}s")
    
    def predict(self, X_test):
        """Make predictions"""
        start_time = time.time()
        predictions = self.model.predict(X_test)
        self.prediction_time = time.time() - start_time
        return predictions
    
    def predict_proba(self, X_test):
        """Get prediction probabilities"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_test)
        return None
    
    def evaluate(self, X_test, y_test):
        """Calculate performance metrics"""
        predictions = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted'),
            'recall': recall_score(y_test, predictions, average='weighted'),
            'f1_score': f1_score(y_test, predictions, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, predictions),
            'training_time': self.training_time,
            'prediction_time': self.prediction_time
        }
        
        # try to calculate ROC AUC
        try:
            proba = self.predict_proba(X_test)
            if proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_test, proba[:, 1])
        except:
            metrics['roc_auc'] = None
        
        return metrics, predictions
    
    def save_model(self, filepath):
        """Save model to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self, filepath):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)


def train_multiple_models(X_train, X_test, y_train, y_test, feature_name):
    """Train all models with given features"""
    print("\n" + "=" * 60)
    print(f"Training models with {feature_name}")
    print("=" * 60)
    
    # models to train
    models_config = {
        'Naive Bayes': {'type': 'naive_bayes', 'params': {}},
        'Logistic Regression': {'type': 'logistic_regression', 'params': {'C': 1.0}},
        'SVM (Linear)': {'type': 'svm', 'params': {'C': 1.0}},
        'Random Forest': {'type': 'random_forest', 'params': {'n_estimators': 100}},
        'KNN': {'type': 'knn', 'params': {'n_neighbors': 5}},
    }
    
    results = {}
    
    for model_name, config in models_config.items():
        print(f"\n{'-' * 60}")
        print(f"{model_name}")
        print(f"{'-' * 60}")
        
        trainer = ModelTrainer(config['type'], **config['params'])
        trainer.train(X_train, y_train)
        metrics, predictions = trainer.evaluate(X_test, y_test)
        
        print(f"\nResults:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        if metrics['roc_auc'] is not None:
            print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        
        results[model_name] = {
            'trainer': trainer,
            'metrics': metrics,
            'predictions': predictions
        }
        
        # save model
        model_dir = f"models/trained/{feature_name.replace(' ', '_').lower()}"
        model_path = os.path.join(model_dir, f"{model_name.replace(' ', '_').lower()}.pkl")
        trainer.save_model(model_path)
    
    return results

def compare_models(all_results):
    """Create comparison table for all models"""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    comparison_data = []
    
    for feature_name, models in all_results.items():
        print(f"\n{feature_name}:")
        print("-" * 60)
        
        for model_name, result in models.items():
            metrics = result['metrics']
            
            row = {
                'Feature Method': feature_name,
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC AUC': metrics['roc_auc'] if metrics['roc_auc'] is not None else 'N/A',
                'Training Time (s)': metrics['training_time']
            }
            
            comparison_data.append(row)
            print(f"  {model_name:25s} - Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
    
    df_comparison = pd.DataFrame(comparison_data)
    
    os.makedirs('results', exist_ok=True)
    df_comparison.to_csv('results/model_comparison.csv', index=False)
    
    print("\n" + "=" * 60)
    print("Best Models:")
    print("=" * 60)
    
    for metric in ['Accuracy', 'F1-Score']:
        if metric == 'ROC AUC':
            df_temp = df_comparison[df_comparison[metric] != 'N/A'].copy()
            df_temp[metric] = pd.to_numeric(df_temp[metric])
            best_row = df_temp.loc[df_temp[metric].idxmax()]
        else:
            best_row = df_comparison.loc[df_comparison[metric].idxmax()]
        
        print(f"\n{metric}:")
        print(f"  Feature: {best_row['Feature Method']}")
        print(f"  Model: {best_row['Model']}")
        print(f"  Score: {best_row[metric]:.4f}")
    
    return df_comparison

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("Model Training")
    print("=" * 60)
    
    print("\nLoading data...")
    y_train = np.load('models/features/y_train.npy')
    y_test = np.load('models/features/y_test.npy')
    
    from feature_extraction import extract_features_multiple_methods, prepare_data
    
    df = pd.read_csv('data/dataset.csv')
    X_train_text, X_test_text, _, _, _ = prepare_data(df)
    features = extract_features_multiple_methods(X_train_text, X_test_text)
    
    all_results = {}
    
    for feature_key, feature_dict in features.items():
        X_train = feature_dict['X_train']
        X_test = feature_dict['X_test']
        feature_name = feature_dict['name']
        
        results = train_multiple_models(X_train, X_test, y_train, y_test, feature_name)
        all_results[feature_name] = results
    
    df_comparison = compare_models(all_results)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("Results saved to results/model_comparison.csv")
    print("=" * 60)
    
    return all_results, df_comparison

if __name__ == "__main__":
    all_results, df_comparison = main()