# Sport vs Politics Text Classifier

Machine learning project to classify news articles as either Sport or Politics.

## Overview

This project compares different text classification approaches:
- **Feature extraction**: Bag of Words, TF-IDF, N-grams
- **ML models**: Naive Bayes, Logistic Regression, SVM, Random Forest, KNN
- **Evaluation**: Accuracy, precision, recall, F1-score, confusion matrices

## Setup

Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk wordcloud
```

Or use requirements file:
```bash
pip install -r requirements.txt
```

## Running the Code

### Complete pipeline:
```bash
python main_pipeline.py
```

### Individual steps:
```bash
python data_collection.py       # get dataset
python feature_extraction.py    # extract features  
python model_training.py        # train models
python evaluation.py            # create plots
```

### Test predictions:
```bash
python predict.py
```

## Project Structure

```
├── data/
│   └── dataset.csv             # collected articles
├── models/
│   ├── features/               # saved vectorizers
│   └── trained/                # trained models
├── results/
│   ├── model_comparison.csv    # all results
│   ├── summary_report.txt      # best models
│   └── plots/                  # visualizations
└── *.py                        # source code
```

## Dataset

The dataset contains news articles from two categories:
- Sport (e.g., football, tennis, olympics)
- Politics (e.g., elections, policy, government)

Source: BBC News dataset + manual collection (~200 samples)

## Feature Extraction

Three approaches tested:
1. **Bag of Words**: Basic word frequency
2. **TF-IDF**: Weighted by importance  
3. **N-grams**: Captures word sequences (bigrams, trigrams)

Preprocessing includes:
- Lowercasing
- Removing special characters
- Tokenization
- Lemmatization
- Stopword removal

## Machine Learning Models

Five classifiers compared:
1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear SVM
4. Random Forest
5. K-Nearest Neighbors

## Results

Check `results/model_comparison.csv` for detailed metrics.

Best performing combination is typically:
- **TF-IDF features** with **Logistic Regression** or **SVM**
- Expected accuracy: 90-95%

## Files

| File | Purpose |
|------|---------|
| data_collection.py | Downloads/creates dataset |
| feature_extraction.py | Extracts BOW, TF-IDF, n-grams |
| model_training.py | Trains all 5 models |
| evaluation.py | Generates plots and metrics |
| main_pipeline.py | Runs everything |
| predict.py | Classify new text |

## Usage Example

```python
from predict import TextClassifier

# load best model
classifier = TextClassifier(
    model_path='models/trained/tfidf_unigrams/logistic_regression.pkl',
    vectorizer_path='models/features/tfidf_vectorizer.pkl'
)

# classify text
text = "The team won the championship match"
prediction, confidence = classifier.predict(text)
print(f"Category: {prediction}")  # Output: Sport
```

## Outputs

The pipeline generates:
- CSV with all model comparisons
- Confusion matrices for each model
- Accuracy/F1-score comparison plots
- Word clouds for both categories

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- nltk
- wordcloud

## Notes

- Dataset size can be adjusted in data_collection.py
- Feature parameters can be modified in feature_extraction.py
- Additional models can be added in model_training.py
