"""
Visualization functions for movie genre prediction
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix
import os

def plot_genre_distribution(df, output_path=None):
    """
    Visualize the distribution of genres
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with a 'MappedGenre' column
    output_path : str, optional
        Path to save the plot
    """
    # Count genres
    mapped_genres = []
    for genres_str in df['MappedGenre']:
        if isinstance(genres_str, str):
            genres = [g.strip() for g in genres_str.split(',')]
            mapped_genres.extend(genres)
    
    mapped_genre_counts = Counter(mapped_genres)
    
    # Plot
    plt.figure(figsize=(12, 8))
    genres = [k for k, v in sorted(mapped_genre_counts.items(), key=lambda x: x[1], reverse=True)]
    counts = [v for k, v in sorted(mapped_genre_counts.items(), key=lambda x: x[1], reverse=True)]
    
    sns.barplot(x=genres, y=counts)
    plt.title('Genre Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Count')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return mapped_genre_counts

def plot_model_comparison(models_results, output_path=None):
    """
    Visualize the performance comparison between models
    
    Parameters:
    -----------
    models_results : dict
        Dictionary with model results
    output_path : str, optional
        Path to save the plot
    """
    # Create comparison dataframe
    comparison_metrics = ['hamming_loss', 'exact_match', 'jaccard_score', 'avg_labels_pred']
    comparison_df = pd.DataFrame(
        {model: [results[metric] for metric in comparison_metrics] 
         for model, results in models_results.items()},
        index=['Hamming Loss', 'Exact Match', 'Jaccard Score', 'Avg Labels Predicted']
    )
    
    # True average labels for reference
    true_avg_labels = models_results[list(models_results.keys())[0]]['avg_labels_true']
    
    # Create subplots for metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Comparison', fontsize=16)
    
    # Hamming Loss (lower is better)
    ax1 = axes[0, 0]
    comparison_df.loc['Hamming Loss'].plot(kind='bar', ax=ax1, color='salmon')
    ax1.set_title('Hamming Loss (lower is better)')
    ax1.set_ylabel('Score')
    for i, v in enumerate(comparison_df.loc['Hamming Loss']):
        ax1.text(i, v + 0.001, f"{v:.4f}", ha='center')
    
    # Exact Match Accuracy (higher is better)
    ax2 = axes[0, 1]
    comparison_df.loc['Exact Match'].plot(kind='bar', ax=ax2, color='skyblue')
    ax2.set_title('Exact Match Accuracy (higher is better)')
    ax2.set_ylabel('Score')
    for i, v in enumerate(comparison_df.loc['Exact Match']):
        ax2.text(i, v + 0.001, f"{v:.4f}", ha='center')
    
    # Jaccard Score (higher is better)
    ax3 = axes[1, 0]
    comparison_df.loc['Jaccard Score'].plot(kind='bar', ax=ax3, color='lightgreen')
    ax3.set_title('Jaccard Score (higher is better)')
    ax3.set_ylabel('Score')
    for i, v in enumerate(comparison_df.loc['Jaccard Score']):
        ax3.text(i, v + 0.001, f"{v:.4f}", ha='center')
    
    # Average Labels
    ax4 = axes[1, 1]
    comparison_df.loc['Avg Labels Predicted'].plot(kind='bar', ax=ax4, color='purple')
    ax4.axhline(y=true_avg_labels, color='r', linestyle='--', 
                label=f'True Avg: {true_avg_labels:.2f}')
    ax4.set_title('Average Labels per Movie')
    ax4.set_ylabel('Count')
    ax4.legend()
    for i, v in enumerate(comparison_df.loc['Avg Labels Predicted']):
        ax4.text(i, v + 0.05, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def plot_confusion_matrices(y_test, y_pred, genre_classes, output_path=None):
    """
    Plot confusion matrices for each genre
    
    Parameters:
    -----------
    y_test : numpy.ndarray
        True labels
    y_pred : numpy.ndarray
        Predicted labels
    genre_classes : list
        List of genre names
    output_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(15, 15))
    
    # Get top genres by frequency
    top_genre_indices = np.argsort(-np.sum(y_test, axis=0))
    
    # Plot confusion matrices for all genres
    num_genres = len(genre_classes)
    rows = int(np.ceil(num_genres / 3))
    
    for i, idx in enumerate(top_genre_indices):
        genre = genre_classes[idx]
        plt.subplot(rows, 3, i+1)
        cm = confusion_matrix(y_test[:, idx], y_pred[:, idx])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Not '+genre, genre],
                   yticklabels=['Not '+genre, genre])
        plt.title(f'Confusion Matrix: {genre}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def plot_genre_cooccurrence(y_train, genre_classes, output_path=None):
    """
    Visualize genre co-occurrence
    
    Parameters:
    -----------
    y_train : numpy.ndarray
        Binary encoded target labels
    genre_classes : list
        List of genre names
    output_path : str, optional
        Path to save the plot
    """
    # Create co-occurrence matrix
    co_occurrence = np.dot(y_train.T, y_train)
    np.fill_diagonal(co_occurrence, 0)  # Remove self-co-occurrences
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(co_occurrence, annot=True, fmt='d', cmap='Greens',
               xticklabels=genre_classes, yticklabels=genre_classes)
    plt.title('Genre Co-occurrence Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Also print top co-occurring pairs
    co_pairs = []
    for i in range(len(genre_classes)):
        for j in range(i+1, len(genre_classes)):
            co_pairs.append((genre_classes[i], genre_classes[j], co_occurrence[i, j]))
    
    # Sort by co-occurrence count (descending)
    co_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print("\nTop Genre Co-occurrences:")
    for genre1, genre2, count in co_pairs[:10]:
        print(f"{genre1} + {genre2}: {count} occurrences")
        
    return co_pairs

def plot_feature_importance(vectorizer, model, genre_classes, top_n=10, output_path=None):
    """
    Visualize the most important features for each genre
    
    Parameters:
    -----------
    vectorizer : TfidfVectorizer
        Fitted vectorizer
    model : sklearn model
        Trained model
    genre_classes : list
        List of genre names
    top_n : int
        Number of top features to plot
    output_path : str, optional
        Path to save the plot
    """
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # For OneVsRestClassifier
    if hasattr(model, 'estimators_'):
        coefficients = []
        for estimator in model.estimators_:
            if hasattr(estimator, 'coef_'):
                coefficients.append(estimator.coef_[0])
            else:
                coefficients.append(np.zeros(len(feature_names)))
        coefficients = np.array(coefficients)
    # For other models
    elif hasattr(model, 'coef_'):
        coefficients = model.coef_
    else:
        print("Model doesn't have interpretable coefficients")
        return
    
    # Plot top features for each genre
    plt.figure(figsize=(15, 20))
    
    for i, genre in enumerate(genre_classes):
        if i >= len(coefficients):
            continue
            
        # Get coefficients for this genre
        coef = coefficients[i]
        
        # Get top positive coefficients
        top_positive_idx = np.argsort(coef)[-top_n:]
        top_features = [(feature_names[idx], coef[idx]) for idx in top_positive_idx]
        top_features.reverse()  # Sort descending
        
        # Plot
        plt.subplot(len(genre_classes), 1, i+1)
        plt.barh([f[0] for f in top_features], [f[1] for f in top_features])
        plt.title(f'Top {top_n} features for {genre}')
        plt.xlabel('Coefficient')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()
