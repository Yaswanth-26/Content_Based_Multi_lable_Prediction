"""
Tests for the text preprocessing module
"""

import unittest
from src.data.preprocessor import preprocess_text, get_pos_distribution

class TestPreprocessor(unittest.TestCase):
    """
    Test case for text preprocessing functions
    """
    
    def test_preprocess_text(self):
        """Test the preprocess_text function"""
        # Test with a sample plot
        sample_plot = "A bartender is working at a saloon, serving drinks to customers."
        tokens = preprocess_text(sample_plot)
        
        # Check if tokens are returned as a list
        self.assertIsInstance(tokens, list)
        
        # Check if basic preprocessing was done
        self.assertIn('bartender', tokens)
        self.assertIn('serve', tokens)  # 'serving' should be lemmatized to 'serve'
        
        # Stopwords should be removed
        self.assertNotIn('is', tokens)
        self.assertNotIn('at', tokens)
        self.assertNotIn('to', tokens)
        
        # Short words should be removed
        self.assertNotIn('a', tokens)
        
    def test_preprocess_text_with_nonstring(self):
        """Test preprocess_text with non-string input"""
        # Test with None
        tokens = preprocess_text(None)
        self.assertEqual(tokens, [])
        
        # Test with a number
        tokens = preprocess_text(123)
        self.assertEqual(tokens, [])
        
    def test_get_pos_distribution(self):
        """Test the get_pos_distribution function"""
        # Test with a sample token list
        tokens = ['man', 'walk', 'fast', 'house']
        pos_dist = get_pos_distribution(tokens)
        
        # Check if all required keys are present
        self.assertIn('noun_pct', pos_dist)
        self.assertIn('verb_pct', pos_dist)
        self.assertIn('adj_pct', pos_dist)
        self.assertIn('adv_pct', pos_dist)
        
        # Check if percentages sum to 1.0 (approximately)
        total = sum(pos_dist.values())
        self.assertAlmostEqual(total, 1.0, places=1)
        
        # Test with empty list
        empty_pos_dist = get_pos_distribution([])
        self.assertEqual(empty_pos_dist['noun_pct'], 0)
        self.assertEqual(empty_pos_dist['verb_pct'], 0)
        self.assertEqual(empty_pos_dist['adj_pct'], 0)
        self.assertEqual(empty_pos_dist['adv_pct'], 0)

if __name__ == '__main__':
    unittest.main()
