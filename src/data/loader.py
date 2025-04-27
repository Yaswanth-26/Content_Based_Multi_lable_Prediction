"""
Functions for loading and exploring the movie dataset
"""

import pandas as pd
import re
from collections import Counter

def load_data(file_path):
    """
    Load the movie dataset from a CSV file
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe
    """
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully with {len(df)} rows and {len(df.columns)} columns")
    return df

def extract_all_genres(df):
    """
    Extract and count all unique genres in the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing a 'Genre' column
        
    Returns:
    --------
    dict
        Dictionary with genre counts
    """
    all_genres = []
    for genre_text in df['Genre']:
        if isinstance(genre_text, str):
            # Split using multiple potential delimiters
            genres = re.split(r'[,|;/]', genre_text)
            # Clean up each genre
            genres = [g.strip() for g in genres if g.strip()]
            all_genres.extend(genres)
            
    genre_counts = Counter(all_genres)
    return genre_counts

def explore_dataset(df):
    """
    Explore the dataset and print basic information
    
    Parameters:
    -----------
    df : pd.DataFrame
        The movie dataframe
        
    Returns:
    --------
    dict
        Dictionary with dataset statistics
    """
    # Extract main columns for classification
    df_main = df[['Plot', 'Genre']].copy()
    
    # Basic information
    stats = {}
    stats['row_count'] = len(df)
    stats['column_count'] = len(df.columns)
    stats['plot_null_count'] = df['Plot'].isnull().sum()
    stats['genre_null_count'] = df['Genre'].isnull().sum()
    
    # Genre statistics
    genre_counts = extract_all_genres(df)
    stats['unique_genres'] = len(genre_counts)
    stats['top_genres'] = [(k, v) for k, v in sorted(
        genre_counts.items(), key=lambda x: x[1], reverse=True)[:10]]
    
    # Plot length statistics
    df['plot_length'] = df['Plot'].apply(lambda x: len(str(x)) if isinstance(x, str) else 0)
    stats['avg_plot_length'] = df['plot_length'].mean()
    stats['min_plot_length'] = df['plot_length'].min()
    stats['max_plot_length'] = df['plot_length'].max()
    
    return stats
