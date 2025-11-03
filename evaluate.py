# Evaluation Script
import json

def evaluate_on_test_set(test_file='test_words.txt'):
    """Evaluate trained agent on test set"""
    print("Loading models...")
    # Load HMM and agent here
    
    with open(test_file, 'r') as f:
        test_words = [line.strip().lower() for line in f if line.strip()]
    
    print(f"Test set size: {len(test_words)} words")
    
    # Run evaluation loop
    # Calculate metrics
    # Save results
    
    return results
