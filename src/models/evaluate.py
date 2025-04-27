"""
Evaluation functions for multi-label classification models
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score,
    f1_score, 
    hamming_loss,
    jaccard_score, 
    classification_report,
    confusion_matrix
)

def evaluate_model(y_true, y_pred, genre_classes, model_name):
    """
    Evaluate model performance with appropriate metrics for multi-label classification
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels
    y_pred : numpy.ndarray
        Predicted labels
    genre_classes : list
        List of genre names
    model_name : str
        Name of the model being evaluated
        
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    # Calculate performance metrics
    hamming = hamming_loss(y_true, y_pred)
    print(f"{model_name} - Hamming Loss: {hamming:.4f}")
    
    # Calculate subset accuracy (exact match)
    subset_accuracy = accuracy_score(y_true, y_pred)
    print(f"{model_name} - Subset Accuracy (Exact Match): {subset_accuracy:.4f}")
    
    # Calculate Jaccard score (similarity coefficient)
    jaccard = jaccard_score(y_true, y_pred, average='samples')
    print(f"{model_name} - Jaccard Score: {jaccard:.4f}")
    
    # Calculate per-class metrics
    print(f"\n{model_name} - Classification Report:")
    report = classification_report(y_true, y_pred, target_names=genre_classes, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=genre_classes))
    
    # Calculate the average number of labels per sample
    avg_labels_pred = np.sum(y_pred, axis=1).mean()
    avg_labels_true = np.sum(y_true, axis=1).mean()
    print(f"{model_name} - Average number of genres per movie (true): {avg_labels_true:.2f}")
    print(f"{model_name} - Average number of genres per movie (predicted): {avg_labels_pred:.2f}")
    
    # Store all metrics in a dictionary
    results = {
        'hamming_loss': hamming,
        'exact_match': subset_accuracy,
        'jaccard_score': jaccard,
        'avg_labels_pred': avg_labels_pred,
        'avg_labels_true': avg_labels_true,
        'report': report
    }
    
    return results

def get_confusion_matrices(y_true, y_pred, genre_classes):
    """
    Calculate confusion matrices for each genre
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels
    y_pred : numpy.ndarray
        Predicted labels
    genre_classes : list
        List of genre names
        
    Returns:
    --------
    dict
        Dictionary of confusion matrices for each genre
    """
    confusion_matrices = {}
    
    for i, genre in enumerate(genre_classes):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        confusion_matrices[genre] = cm
        
    return confusion_matrices

def get_per_genre_metrics(y_true, y_pred, genre_classes):
    """
    Calculate precision, recall and F1 score for each genre
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels
    y_pred : numpy.ndarray
        Predicted labels
    genre_classes : list
        List of genre names
        
    Returns:
    --------
    dict
        Dictionary with metrics for each genre
    """
    metrics = {}
    
    for i, genre in enumerate(genre_classes):
        precision = precision_score(y_true[:, i], y_pred[:, i])
        recall = recall_score(y_true[:, i], y_pred[:, i])
        f1 = f1_score(y_true[:, i], y_pred[:, i])
        
        metrics[genre] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
    return metrics

def predict_example_plots(model, vectorizer, example_plots, genre_classes, threshold=0.3):
    """
    Make predictions on example plot descriptions
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    vectorizer : TfidfVectorizer
        Fitted vectorizer
    example_plots : list
        List of plot descriptions
    genre_classes : list
        List of genre classes
    threshold : float
        Probability threshold for prediction
        
    Returns:
    --------
    list
        List of predictions with probabilities
    """
    from src.data.preprocessor import preprocess_text
    
    results = []
    
    for plot in example_plots:
        # Preprocess text
        tokens = preprocess_text(plot)
        processed_text = ' '.join(tokens)
        
        # Transform to TF-IDF
        X = vectorizer.transform([processed_text])
        
        # For models with predict_proba
        if hasattr(model, 'predict_proba'):
            # Handle OneVsRestClassifier differently from other models
            if hasattr(model, 'estimators_'):  # MultiOutputClassifier
                probas = []
                for i, estimator in enumerate(model.estimators_):
                    if i < len(genre_classes):
                        proba = estimator.predict_proba(X)[0]
                        if len(proba) > 1:  # Binary classification
                            probas.append(proba[1])  # Probability of positive class
                        else:
                            probas.append(0.0)
                    else:
                        probas.append(0.0)
                probas = np.array(probas)
            else:  # OneVsRestClassifier or other
                probas = model.predict_proba(X)[0]
            
            # Get predictions based on threshold
            pred_genres = [(genre_classes[i], float(probas[i])) 
                          for i in range(len(genre_classes)) 
                          if probas[i] >= threshold]
            
        else:  # For models without predict_proba
            y_pred = model.predict(X)[0]
            pred_genres = [(genre_classes[i], 1.0) 
                          for i in range(len(genre_classes)) 
                          if y_pred[i] == 1]
        
        # Sort by probability (descending)
        pred_genres.sort(key=lambda x: x[1], reverse=True)
        
        results.append({
            'plot': plot,
            'predictions': pred_genres
        })
        
    return results
