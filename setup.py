"""
Setup script to create the necessary directory structure for the project
and download required NLTK resources
"""

import os
import sys
import nltk

def create_directories():
    """
    Create the necessary directory structure for the project
    """
    directories = [
        'data/raw',
        'data/processed',
        'output/models',
        'output/plots',
        'notebooks'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def download_nltk_resources():
    """
    Download necessary NLTK resources
    """
    print("\nDownloading NLTK resources...")
    
    resources = [
        'punkt',
        'wordnet',
        'omw-1.4',  # Open Multilingual WordNet
        'stopwords',
        'averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng'  # Added this newer version
    ]
    
    for resource in resources:
        try:
            print(f"Downloading {resource}...")
            nltk.download(resource)
            print(f"Successfully downloaded {resource}")
        except Exception as e:
            print(f"Error downloading {resource}: {str(e)}")

def main():
    """
    Main function to set up the project
    """
    print("Setting up movie genre prediction project...\n")
    
    # Create directories
    create_directories()
    
    # Download NLTK resources
    download_nltk_resources()
    
    print("\nSetup completed successfully!")
    print("Next steps:")
    print("1. Place your dataset in the data/raw/ directory")
    print("2. Install dependencies with: pip install -r requirements.txt")
    print("3. Run the pipeline with: python main.py --data_path data/raw/dataset.csv")

if __name__ == "__main__":
    main()