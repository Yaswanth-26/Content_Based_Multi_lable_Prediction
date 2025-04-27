"""
Text preprocessing functions for movie plot descriptions and genre standardization
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from collections import Counter

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    print("Downloading required NLTK resources...")
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')  # Open Multilingual WordNet
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    try:
        nltk.download('averaged_perceptron_tagger_eng')
    except:
        print("Note: 'averaged_perceptron_tagger_eng' might not be available.")
        print("Using direct download workaround...")
        
        # Try to use the non-eng version and specify language
        def pos_tag_wrapper(tokens):
            """Wrapper for pos_tag that specifies English language"""
            return pos_tag(tokens, lang='en')
        
        # Override the original pos_tag with our wrapper
        globals()['original_pos_tag'] = pos_tag
        globals()['pos_tag'] = pos_tag_wrapper

def get_wordnet_pos(tag):
    """
    Map POS tag to first character used by WordNetLemmatizer
    
    Parameters:
    -----------
    tag : str
        POS tag from nltk.pos_tag
        
    Returns:
    --------
    str
        WordNet POS tag
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        # Default to noun
        return wordnet.NOUN

def preprocess_text(text):
    """
    Comprehensive text preprocessing with POS tagging and lemmatization
    
    Parameters:
    -----------
    text : str
        Raw text to preprocess
        
    Returns:
    --------
    list
        List of preprocessed tokens
    """
    if not isinstance(text, str):
        return []
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # POS tagging
    tagged_tokens = pos_tag(tokens)
    
    # Lemmatization with POS tags
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in tagged_tokens
    ]
    
    # Remove short words (length < 3)
    filtered_tokens = [word for word in lemmatized_tokens if len(word) > 2]
    
    return filtered_tokens

def get_pos_distribution(tokens_list):
    """
    Calculate the distribution of POS tags in a list of tokens
    
    Parameters:
    -----------
    tokens_list : list
        List of tokens
        
    Returns:
    --------
    dict
        Dictionary with POS distribution percentages
    """
    if not tokens_list:
        return {'noun_pct': 0, 'verb_pct': 0, 'adj_pct': 0, 'adv_pct': 0}
    
    tagged = pos_tag(tokens_list)
    total = len(tagged)
    
    pos_counts = {
        'noun_pct': sum(1 for _, tag in tagged if tag.startswith('N')) / total,
        'verb_pct': sum(1 for _, tag in tagged if tag.startswith('V')) / total,
        'adj_pct': sum(1 for _, tag in tagged if tag.startswith('J')) / total,
        'adv_pct': sum(1 for _, tag in tagged if tag.startswith('R')) / total
    }
    
    return pos_counts

def filter_rare_genres(genre_text, popular_genres, min_count=100):
    """
    Filter out rare genres
    
    Parameters:
    -----------
    genre_text : str
        Comma-separated genre string
    popular_genres : set
        Set of popular genres to keep
        
    Returns:
    --------
    str or None
        Filtered genre string, or None if all genres filtered out
    """
    if not isinstance(genre_text, str):
        return genre_text
    
    # Split and clean genres
    genres = re.split(r'[,|;/]', genre_text)
    genres = [g.strip() for g in genres if g.strip()]
    
    # Keep only popular genres
    filtered_genres = [g for g in genres if g in popular_genres]
    
    # Return comma-separated string of filtered genres
    if filtered_genres:
        return ', '.join(filtered_genres)
    else:
        return None  # Will be handled as NaN and can be dropped

def standardize_genres(df, min_genre_count=100):
    """
    Standardize genres by removing rare ones and mapping to standard categories
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with a 'Genre' column
    min_genre_count : int
        Minimum occurrences to consider a genre
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with standardized genres
    """
    # Extract all genres and count frequencies
    all_genres = []
    for genre_text in df['Genre']:
        if isinstance(genre_text, str):
            genres = re.split(r'[,|;/]', genre_text)
            genres = [g.strip() for g in genres if g.strip()]
            all_genres.extend(genres)
    
    genre_counts = Counter(all_genres)
    
    # Keep only genres that appear at least min_genre_count times
    popular_genres = {genre for genre, count in genre_counts.items() 
                     if count >= min_genre_count}
    
    # Apply filtering
    df['Genre'] = df['Genre'].apply(lambda x: filter_rare_genres(x, popular_genres))
    
    # Remove rows where all genres were filtered out
    df_filtered = df.dropna(subset=['Genre'])
    
    # Define major genres to keep
    major_genres = [
        'drama', 'comedy', 'romance', 'action', 'crime', 'horror', 'thriller',
        'science fiction', 'musical', 'mystery', 'western', 'adventure',
        'historical', 'war', 'animation', 'suspense', 'biography',
        'social', 'family', 'fantasy'
    ]
    
    # Define genre mapping
    genre_mapping = {
        'sci-fi': 'science fiction',
        'film noir': 'thriller',
        'animated': 'animation',
        'documentary': 'drama',
        'noir film': 'thriller'
    }
    
    # Some genres map to multiple main genres
    multiple_mappings = {
        'romantic comedy': ['comedy', 'romance'],
        'comedy-drama': ['comedy', 'drama'],
        'comedy drama': ['comedy', 'drama'],
        'crime drama': ['crime', 'drama'],
        'romantic drama': ['romance', 'drama'],
        'musical comedy': ['musical', 'comedy']
    }
    
    # List of genres to remove completely
    genres_to_remove = ['unknown']
    
    # Create a new dataframe to hold the standardized data
    df_standardized = pd.DataFrame(columns=['Plot', 'Genre'])
    
    # Process each row in the filtered dataframe
    for index, row in df_filtered.iterrows():
        plot = row['Plot']
        genres_str = row['Genre']
        
        if not isinstance(genres_str, str):
            continue
            
        # Split genres
        genres = [g.strip() for g in re.split(r'[,|;/]', genres_str)]
        
        # Remove unwanted genres
        genres = [g for g in genres if g not in genres_to_remove]
        
        # If no genres left after removal, skip this row
        if not genres:
            continue
            
        # Map to standard genres
        standard_genres = set()
        for genre in genres:
            # Check if this genre maps to multiple main genres
            if genre in multiple_mappings:
                standard_genres.update(multiple_mappings[genre])
            # Check if this genre maps to a single main genre
            elif genre in genre_mapping:
                standard_genres.add(genre_mapping[genre])
            # Otherwise, keep the original genre if it's in our major genres list
            elif genre in major_genres:
                standard_genres.add(genre)
                
        # If we have standard genres, add this to our standardized dataframe
        if standard_genres:
            standard_genres_str = ', '.join(sorted(standard_genres))
            df_standardized = df_standardized._append({'Plot': plot, 'Genre': standard_genres_str}, ignore_index=True)
    
    # Second-level genre mapping for final consolidation
    final_genre_mapping = {
        # Map to drama
        'biography': 'drama',
        'historical': 'drama',
        'social': 'drama',
        'war': 'drama',
        
        # Map to action
        'adventure': 'action',
        'western': 'action',
        
        # Map to thriller
        'mystery': 'thriller',
        'suspense': 'thriller',
        
        # Map to comedy
        'musical': 'comedy',
        
        # Map to other categories
        'science fiction': 'fantasy',  # Combine sci-fi and fantasy
        'animation': 'family'  # Animation is often family-friendly
    }
    
    # Apply final mapping
    def apply_mapping(genre, mapping):
        if genre in mapping:
            return mapping[genre]
        return genre
        
    df_standardized['MappedGenre'] = df_standardized['Genre'].apply(
        lambda x: ', '.join([apply_mapping(g.strip(), final_genre_mapping) for g in x.split(',')]) 
        if isinstance(x, str) else x
    )
    
    for col in ['processed_plots', 'processed_text']:
        if col in df.columns and col not in df_standardized.columns:
           df_standardized[col] = df[col]

    return df_standardized