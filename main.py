"""
Movie Genre Prediction - Main Script
Run this script to execute the complete machine learning pipeline from data loading to model evaluation.
"""

import argparse
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

# Import modules from src
from src.data.loader import load_data
from src.data.preprocessor import preprocess_text, standardize_genres
from src.features.text_features import create_tfidf_features
from src.models.train import (
    train_one_vs_rest_model,
    train_multi_output_model,
    train_classifier_chain_model
)
from src.models.evaluate import evaluate_model
from src.visualization.plots import (
    plot_genre_distribution,
    plot_model_comparison,
    plot_confusion_matrices,
    plot_genre_cooccurrence
)

def main(args):
    """
    Main function to run the complete machine learning pipeline
    """
    # Check if data directory exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        print("Make sure to create the data/raw directory and place your dataset there.")
        print("You can run 'python setup.py' to create the necessary directories.")
        return
        
    # Create output directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)
    
    # Create data/processed directory if it doesn't exist
    data_dir = os.path.dirname(args.data_path)
    processed_dir = os.path.join(os.path.dirname(data_dir), "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    print("1. Loading data...")
    df = load_data(args.data_path)
    
    print("2. Preprocessing text and standardizing genres...")
    df_processed = df[['Plot', 'Genre']].copy()
    
    # Apply text preprocessing
    print("   - Preprocessing plot descriptions...")
    df_processed['processed_plots'] = df_processed['Plot'].apply(preprocess_text)
    df_processed['processed_text'] = df_processed['processed_plots'].apply(lambda tokens: ' '.join(tokens))
    
    # Standardize genres
    print("   - Standardizing genres...")
    df_standardized = standardize_genres(df_processed)
    if 'processed_text' not in df_standardized.columns and 'processed_plots' in df_standardized.columns:
        df_standardized['processed_text'] = df_standardized['processed_plots'].apply(lambda tokens: ' '.join(tokens))
    # Plot genre distribution
    print("   - Plotting genre distribution...")
    plot_genre_distribution(df_standardized, os.path.join(args.output_dir, "plots", "genre_distribution.png"))
    
    # Split into features and target
    print("3. Creating features and preparing datasets...")
    X = df_standardized['processed_text']
    
    # Convert genres to binary format
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    
    def get_genre_list(genre_string):
        if isinstance(genre_string, str):
            return [genre.strip() for genre in genre_string.split(',')]
        return []
    
    genre_lists = df_standardized['MappedGenre'].apply(get_genre_list)
    y_encoded = mlb.fit_transform(genre_lists)
    genre_classes = mlb.classes_
    
    print(f"   - Found {len(genre_classes)} genres: {genre_classes}")
    
    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=args.test_size, random_state=args.random_state
    )
    
    print(f"   - Training set size: {len(X_train)}")
    print(f"   - Test set size: {len(X_test)}")
    
    # Feature engineering with TF-IDF
    print("4. Creating TF-IDF features...")
    vectorizer, X_train_tfidf, X_test_tfidf = create_tfidf_features(
        X_train, X_test, max_features=args.max_features
    )
    
    # Train models
    print("5. Training models...")
    print("   - Training One-vs-Rest classifier...")
    ovr_clf = train_one_vs_rest_model(X_train_tfidf, y_train)
    
    print("   - Training Multi-Output classifier...")
    multi_clf = train_multi_output_model(X_train_tfidf, y_train)
    
    print("   - Training Classifier Chain model...")
    chain_clf = train_classifier_chain_model(X_train_tfidf, y_train, genre_classes)
    
    # Make predictions
    print("6. Evaluating models...")
    y_pred_ovr = ovr_clf.predict(X_test_tfidf)
    y_pred_multi = multi_clf.predict(X_test_tfidf)
    y_pred_chain = chain_clf.predict(X_test_tfidf)
    
    # Evaluate models
    print("   - Evaluating One-vs-Rest model...")
    ovr_results = evaluate_model(y_test, y_pred_ovr, genre_classes, "One-vs-Rest")
    
    print("   - Evaluating Multi-Output model...")
    multi_results = evaluate_model(y_test, y_pred_multi, genre_classes, "Multi-Output")
    
    print("   - Evaluating Classifier Chain model...")
    chain_results = evaluate_model(y_test, y_pred_chain, genre_classes, "Classifier Chain")
    
    # Visualize results
    print("7. Visualizing results...")
    models_results = {
        'One-vs-Rest': ovr_results,
        'Multi-Output': multi_results,
        'Classifier Chain': chain_results
    }
    
    # Plot model comparison
    plot_model_comparison(
        models_results, 
        os.path.join(args.output_dir, "plots", "model_comparison.png")
    )
    
    # Determine best model
    best_model_name = min(models_results, key=lambda m: models_results[m]['hamming_loss'])
    print(f"\nBest model based on Hamming Loss: {best_model_name}")
    
    # Plot confusion matrices for best model
    if best_model_name == 'One-vs-Rest':
        y_pred_best = y_pred_ovr
        best_model = ovr_clf
    elif best_model_name == 'Multi-Output':
        y_pred_best = y_pred_multi
        best_model = multi_clf
    else:
        y_pred_best = y_pred_chain
        best_model = chain_clf
    
    plot_confusion_matrices(
        y_test, y_pred_best, genre_classes,
        os.path.join(args.output_dir, "plots", "confusion_matrices.png")
    )
    
    # Plot genre co-occurrence
    plot_genre_cooccurrence(
        y_train, genre_classes,
        os.path.join(args.output_dir, "plots", "genre_cooccurrence.png")
    )
    
    # Save models and components
    print("8. Saving models and components...")
    joblib.dump(ovr_clf, os.path.join(args.output_dir, "models", "onevsrest_genre_classifier.pkl"))
    joblib.dump(multi_clf, os.path.join(args.output_dir, "models", "multioutput_genre_classifier.pkl"))
    joblib.dump(chain_clf, os.path.join(args.output_dir, "models", "classifier_chain_genre_model.pkl"))
    joblib.dump(vectorizer, os.path.join(args.output_dir, "models", "tfidf_vectorizer.pkl"))
    joblib.dump(mlb, os.path.join(args.output_dir, "models", "genre_binarizer.pkl"))
    
    # Save the best model separately for easier reference
    joblib.dump(best_model, os.path.join(args.output_dir, "models", "best_model.pkl"))
    
    print("\nPipeline completed successfully!")
    print(f"Results saved to {args.output_dir}")
    print(f"Best model: {best_model_name}")
    
    best_metrics = models_results[best_model_name]
    print(f"   - Hamming Loss: {best_metrics['hamming_loss']:.4f}")
    print(f"   - Exact Match Accuracy: {best_metrics['exact_match']:.4f}")
    print(f"   - Jaccard Score: {best_metrics['jaccard_score']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Movie Genre Prediction Pipeline")
    
    parser.add_argument("--data_path", type=str, default="data/raw/dataset.csv",
                        help="Path to the raw dataset CSV file")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save output files")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Proportion of data to use for testing")
    parser.add_argument("--max_features", type=int, default=10000,
                        help="Maximum number of features for TF-IDF vectorizer")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random state for reproducibility")
    
    args = parser.parse_args()
    main(args)