"""
Example script to demonstrate genre prediction on new movie plots
"""

import sys
import os

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.predict import predict_genres

def main():
    """
    Main function to demonstrate genre prediction
    """
    # Example movie plots
    example_plots = [
        "A detective investigates a series of murders in a small town.",
        "Two people from different social backgrounds fall in love despite their families' objections.",
        "A spaceship crew discovers an alien life form that starts killing them one by one.",
        "A comedian struggles with addiction while trying to revive his failing career.",
        "A group of friends plan the perfect heist to rob a bank."
    ]
    
    print("Movie Genre Prediction Example\n")
    print("Using the trained model to predict genres for example plots:\n")
    
    for i, plot in enumerate(example_plots):
        print(f"Example {i+1}: {plot}")
        predictions = predict_genres(plot)
        
        if predictions:
            print("Predicted genres:")
            for genre, prob in predictions:
                print(f"  {genre}: {prob:.4f}")
        else:
            print("No genres predicted above threshold.")
        print()

if __name__ == "__main__":
    main()
