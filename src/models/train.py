"""
Model training functions for multi-label classification
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import ClassifierChain

def train_one_vs_rest_model(X_train, y_train, C=1.0):
    """
    Train a One-vs-Rest classifier for multi-label classification
    
    Parameters:
    -----------
    X_train : scipy.sparse.csr.csr_matrix
        Training features
    y_train : numpy.ndarray
        Binary encoded target labels
    C : float
        Regularization parameter
        
    Returns:
    --------
    OneVsRestClassifier
        Trained model
    """
    # Create base classifier
    base_clf = LogisticRegression(C=C, solver='liblinear', max_iter=1000)
    
    # Create and train One-vs-Rest classifier
    ovr_clf = OneVsRestClassifier(base_clf, n_jobs=-1)
    ovr_clf.fit(X_train, y_train)
    
    return ovr_clf

def train_multi_output_model(X_train, y_train, C=1.0):
    """
    Train a Multi-Output classifier for multi-label classification
    
    Parameters:
    -----------
    X_train : scipy.sparse.csr.csr_matrix
        Training features
    y_train : numpy.ndarray
        Binary encoded target labels
    C : float
        Regularization parameter
        
    Returns:
    --------
    MultiOutputClassifier
        Trained model
    """
    # Create base classifier
    base_clf = LogisticRegression(C=C, solver='liblinear', max_iter=1000)
    
    # Create and train multi-label classifier
    multi_label_clf = MultiOutputClassifier(base_clf, n_jobs=-1)
    multi_label_clf.fit(X_train, y_train)
    
    return multi_label_clf

def train_classifier_chain_model(X_train, y_train, genre_classes, C=1.0):
    """
    Train a Classifier Chain model for multi-label classification
    
    Parameters:
    -----------
    X_train : scipy.sparse.csr.csr_matrix
        Training features
    y_train : numpy.ndarray
        Binary encoded target labels
    genre_classes : list
        List of genre classes
    C : float
        Regularization parameter
        
    Returns:
    --------
    ClassifierChain
        Trained model
    """
    # Calculate genre frequencies in training data
    genre_frequencies = y_train.sum(axis=0)
    
    # Sort genres by frequency (most common first)
    chain_order = np.argsort(-genre_frequencies)
    
    print("Chain order based on genre frequency:")
    for i, idx in enumerate(chain_order):
        print(f"{i+1}. {genre_classes[idx]} ({genre_frequencies[idx]} occurrences)")
    
    # Create base classifier with balanced class weights
    base_clf = LogisticRegression(
        C=C,
        solver='liblinear',
        max_iter=1000,
        class_weight='balanced'
    )
    
    # Create and train classifier chain
    chain_clf = ClassifierChain(
        base_clf,
        order=chain_order,
        random_state=42
    )
    chain_clf.fit(X_train, y_train)
    
    return chain_clf
