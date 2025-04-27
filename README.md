# Movie Genre Prediction

A machine learning system that automatically predicts movie genres based on plot descriptions using multi-label text classification.

## Project Overview

This project addresses the challenge of automatically predicting movie genres based on plot descriptions using natural language processing and multi-label text classification techniques. By analyzing a dataset of 34,886 movies, the system demonstrates the efficacy of machine learning approaches in categorizing films across multiple genres simultaneously.

### Features

- Text preprocessing pipeline (tokenization, lemmatization, POS tagging)
- TF-IDF vectorization for feature extraction
- Multiple multi-label classification models:
  - One-vs-Rest Logistic Regression
  - Multi-Output Classification
  - Classifier Chain
- Genre co-occurrence analysis
- Model evaluation with appropriate multi-label metrics

## Dataset

The project uses a dataset containing 34,886 movie entries with plot descriptions and genre labels. After preprocessing and standardization, the final dataset contains 25,381 movies categorized across 9 major genres:
- Drama
- Comedy
- Action
- Romance
- Thriller
- Crime
- Horror
- Fantasy
- Family

## Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/movie-genre-prediction.git
cd movie-genre-prediction


# Install dependencies
pip install -r requirements.txt

# Place your dataset in data/raw/
# For example: data/raw/dataset.csv
```

## Usage

### Training the models

```bash
# Run the complete pipeline
python main.py

# Optionally specify parameters
python main.py --data_path data/raw/dataset.csv --test_size 0.2
```

### Using pre-trained models for prediction

```python
from src.models.predict import predict_genres

# Example
plot = "A detective investigates a series of murders in a small town."
genres = predict_genres(plot)
print(genres)
# Output: [('crime', 0.3747)]
```

## Project Structure

```
movie-genre-prediction/
├── README.md                      # Project overview and instructions
├── requirements.txt               # Dependencies
├── data/                          # Data files
├── notebooks/                     # Jupyter notebooks
├── src/                           # Source code modules
│   ├── data/                      # Data processing
│   ├── features/                  # Feature engineering
│   ├── models/                    # ML models
│   └── visualization/             # Visualization tools
├── tests/                         # Unit tests
└── main.py                        # Main script to run pipeline
```

## Results

The final model achieved:
- Hamming Loss: 0.1004
- Exact Match Accuracy: 0.3514
- Jaccard Score: 0.4041

Analysis revealed distinctive linguistic patterns associated with different genres and identified common genre co-occurrences such as comedy-drama, comedy-romance, and crime-drama.

## License

[MIT](LICENSE)