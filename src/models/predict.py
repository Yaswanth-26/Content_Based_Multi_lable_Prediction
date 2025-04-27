"""
Functions for making predictions with trained models
"""

import os
import joblib
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import ClassifierChain

def load_models(models_dir="output/models"):
    """
    Load all trained models and components
    
    Parameters:
    -----------
    models_dir : str
        Directory containing saved models
        
    Returns:
    --------
    dict
        Dictionary with loaded models and components
    """
    models = {}
    
    models['vectorizer'] = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.pkl"))
    models['binarizer'] = joblib.load(os.path.join(models_dir, "genre_binarizer.pkl"))
    models['best_model'] = joblib.load(os.path.join(models_dir, "best_model.pkl"))
    
    # Optionally load other models
    try:
        models['one_vs_rest'] = joblib.load(os.path.join(models_dir, "onevsrest_genre_classifier.pkl"))
        models['multi_output'] = joblib.load(os.path.join(models_dir, "multioutput_genre_classifier.pkl"))
        models['chain'] = joblib.load(os.path.join(models_dir, "classifier_chain_genre_model.pkl"))
    except:
        pass
        
    return models

def predict_genres(plot_text, models_dir="output/models", threshold=0.3):
    """
    Predict genres for a new plot text using the best trained model
    
    Parameters:
    -----------
    plot_text : str
        Plot description text
    models_dir : str
        Directory containing saved models
    threshold : float
        Probability threshold for prediction
        
    Returns:
    --------
    list
        List of (genre, probability) tuples
    """
    from src.data.preprocessor import preprocess_text
    
    # Load models
    models = load_models(models_dir)
    vectorizer = models['vectorizer']
    binarizer = models['binarizer']
    model = models['best_model']
    genre_classes = binarizer.classes_
    
    # Preprocess the text
    tokens = preprocess_text(plot_text)
    processed_text = ' '.join(tokens)
    
    # Transform the text to TF-IDF features
    X = vectorizer.transform([processed_text])
    
    # For models without predict_proba, just use predict
    if not hasattr(model, 'predict_proba'):
        y_pred = model.predict(X)[0]
        predictions = [(genre, 1.0) for i, genre in enumerate(genre_classes) if y_pred[i] == 1]
        return predictions
    
    # Handling for different model types with predict_proba
    predictions = []
    
    # Special handling for OneVsRestClassifier
    if isinstance(model, OneVsRestClassifier):
        # OneVsRest returns decision function scores or probability array
        y_scores = model.predict_proba(X)[0]
        for i, score in enumerate(y_scores):
            if score >= threshold:
                predictions.append((genre_classes[i], float(score)))
                
    # Special handling for MultiOutputClassifier
    elif isinstance(model, MultiOutputClassifier):
        # MultiOutputClassifier returns a list of probability arrays for each label
        for i, estimator in enumerate(model.estimators_):
            if i < len(genre_classes):  # Safety check
                proba = estimator.predict_proba(X)[0]
                if len(proba) > 1:  # Binary classification
                    score = proba[1]  # Probability of positive class
                    if score >= threshold:
                        predictions.append((genre_classes[i], float(score)))
                        
    # Special handling for ClassifierChain
    elif isinstance(model, ClassifierChain):
        # First get the binary predictions
        y_pred = model.predict(X)[0]
        # For classifier chain, we don't have direct access to probabilities for each class
        # So we'll use the binary predictions with a default score
        predictions = [(genre, 0.8) for i, genre in enumerate(genre_classes) if y_pred[i] == 1]
        
    # Generic fallback - just use predict
    else:
        y_pred = model.predict(X)[0]
        predictions = [(genre, 1.0) for i, genre in enumerate(genre_classes) if y_pred[i] == 1]
    
    # Sort by probability (descending)
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    return predictions

def predict_batch(plot_texts, models_dir="output/models", threshold=0.3):
    """
    Predict genres for multiple plot texts
    
    Parameters:
    -----------
    plot_texts : list
        List of plot description texts
    models_dir : str
        Directory containing saved models
    threshold : float
        Probability threshold for prediction
        
    Returns:
    --------
    list
        List of prediction results, each containing the plot and predicted genres
    """
    results = []
    
    for plot in plot_texts:
        predictions = predict_genres(plot, models_dir, threshold)
        results.append({
            'plot': plot,
            'predictions': predictions
        })
        
    return results
