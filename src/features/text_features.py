"""
Feature engineering functions for text data
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from collections import Counter

def create_tfidf_features(X_train, X_test, max_features=10000, ngram_range=(1, 2)):
    """
    Create TF-IDF features from text data
    
    Parameters:
    -----------
    X_train : pd.Series
        Training text data
    X_test : pd.Series
        Test text data
    max_features : int
        Maximum number of features to extract
    ngram_range : tuple
        Range of n-grams to include
        
    Returns:
    --------
    tuple
        (vectorizer, X_train_tfidf, X_test_tfidf)
    """
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=5,
        max_df=0.7,
        ngram_range=ngram_range,
        sublinear_tf=True
    )
    
    # Transform training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Transform test data
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"TF-IDF features shape: {X_train_tfidf.shape}")
    
    return vectorizer, X_train_tfidf, X_test_tfidf

def analyze_important_features(vectorizer, X_tfidf, y, top_n=10):
    """
    Analyze the most important features (words) for each genre
    
    Parameters:
    -----------
    vectorizer : TfidfVectorizer
        Fitted vectorizer
    X_tfidf : scipy.sparse.csr.csr_matrix
        TF-IDF transformed features
    y : numpy.ndarray
        Binary encoded target labels
    top_n : int
        Number of top features to return for each class
        
    Returns:
    --------
    dict
        Dictionary with top features for each genre
    """
    feature_names = vectorizer.get_feature_names_out()
    n_classes = y.shape[1]
    
    # Dictionary to store top features for each genre
    top_features = {}
    
    # For each genre
    for genre_idx in range(n_classes):
        # Get positive and negative examples
        positive_indices = np.where(y[:, genre_idx] == 1)[0]
        negative_indices = np.where(y[:, genre_idx] == 0)[0]
        
        # Calculate average TF-IDF for positive and negative examples
        positive_tfidf = X_tfidf[positive_indices].mean(axis=0)
        negative_tfidf = X_tfidf[negative_indices].mean(axis=0)
        
        # Convert to 1D arrays
        positive_tfidf = np.asarray(positive_tfidf).flatten()
        negative_tfidf = np.asarray(negative_tfidf).flatten()
        
        # Calculate difference (importance)
        importance = positive_tfidf - negative_tfidf
        
        # Get indices of top features
        top_indices = np.argsort(importance)[-top_n:]
        
        # Get feature names and importance scores
        top_features_list = [(feature_names[i], importance[i]) for i in top_indices]
        top_features_list.sort(key=lambda x: x[1], reverse=True)
        
        # Store in dictionary
        top_features[genre_idx] = top_features_list
    
    return top_features

def extract_pos_features(processed_plots):
    """
    Extract Part-of-Speech distribution features
    
    Parameters:
    -----------
    processed_plots : pd.Series
        Series containing lists of preprocessed tokens
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with POS distribution features
    """
    from src.data.preprocessor import get_pos_distribution
    
    pos_features = processed_plots.apply(get_pos_distribution)
    pos_df = pd.DataFrame(pos_features.tolist())
    
    print(f"POS features shape: {pos_df.shape}")
    
    return pos_df

def get_word_count_features(processed_plots):
    """
    Create features based on word counts and document length
    
    Parameters:
    -----------
    processed_plots : pd.Series
        Series containing lists of preprocessed tokens
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with word count features
    """
    # Word count (document length)
    word_counts = processed_plots.apply(len)
    
    # Unique word count
    unique_word_counts = processed_plots.apply(lambda x: len(set(x)))
    
    # Lexical diversity (unique words / total words)
    lexical_diversity = unique_word_counts / word_counts.apply(lambda x: max(x, 1))
    
    # Create DataFrame
    word_count_df = pd.DataFrame({
        'word_count': word_counts,
        'unique_word_count': unique_word_counts,
        'lexical_diversity': lexical_diversity
    })
    
    return word_count_df
