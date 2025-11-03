# Hidden Markov Model Implementation for Hangman
import numpy as np
from collections import defaultdict, Counter
import pickle

class HangmanHMM:
    """
    HMM trained on word corpus to predict letter probabilities
    Trains separate models for each word length
    """
    
    def __init__(self):
        self.models = {}  # Dictionary to store HMMs by word length
        self.letter_frequencies = {}  # Overall letter frequencies by length
        
    def train(self, corpus_file='corpus.txt'):
        """Train HMM on corpus"""
        print("Loading corpus...")
        with open(corpus_file, 'r') as f:
            words = [line.strip().lower() for line in f if line.strip()]
        
        # Group words by length
        words_by_length = defaultdict(list)
        for word in words:
            if word.isalpha():  # Only alphabetic words
                words_by_length[len(word)].append(word)
        
        print(f"Training HMMs for {len(words_by_length)} different word lengths...")
        
        # Train separate HMM for each word length
        for length, word_list in words_by_length.items():
            print(f"Training length {length}: {len(word_list)} words")
            self.models[length] = self._train_length_model(word_list, length)
            self.letter_frequencies[length] = self._calculate_frequencies(word_list)
        
        print("Training complete!")
    
    def _train_length_model(self, words, length):
        """Train HMM for specific word length"""
        # Position-based emission probabilities
        emission_counts = [defaultdict(int) for _ in range(length)]
        position_totals = [0] * length
        
        for word in words:
            # Count emissions (letter at each position)
            for pos, letter in enumerate(word):
                emission_counts[pos][letter] += 1
                position_totals[pos] += 1
        
        # Convert counts to probabilities
        emission_probs = []
        for pos in range(length):
            pos_probs = {}
            for letter, count in emission_counts[pos].items():
                pos_probs[letter] = count / position_totals[pos]
            emission_probs.append(pos_probs)
        
        return {
            'emission_probs': emission_probs,
            'words': words
        }
    
    def _calculate_frequencies(self, words):
        """Calculate overall letter frequencies for fallback"""
        all_letters = ''.join(words)
        total = len(all_letters)
        freq = Counter(all_letters)
        return {letter: count / total for letter, count in freq.items()}
    
    def predict_letter_probabilities(self, masked_word, guessed_letters):
        """
        Predict probability distribution over remaining letters
        """
        length = len(masked_word)
        
        if length not in self.models:
            return self._fallback_probabilities(guessed_letters, length)
        
        model = self.models[length]
        remaining_letters = set('abcdefghijklmnopqrstuvwxyz') - guessed_letters
        
        # Filter words that match the pattern
        matching_words = self._filter_matching_words(
            model['words'], masked_word, guessed_letters
        )
        
        if len(matching_words) < 5:
            letter_scores = self._position_based_scoring(
                masked_word, model['emission_probs'], remaining_letters
            )
        else:
            letter_scores = self._frequency_based_scoring(
                matching_words, masked_word, remaining_letters
            )
        
        # Normalize to probabilities
        total = sum(letter_scores.values())
        if total == 0:
            return self._fallback_probabilities(guessed_letters, length)
        
        return {letter: score / total for letter, score in letter_scores.items()}
    
    def _filter_matching_words(self, words, masked_word, guessed_letters):
        """Filter words that match the current pattern"""
        matching = []
        for word in words:
            if len(word) != len(masked_word):
                continue
            
            match = True
            for i, char in enumerate(masked_word):
                if char != '_':
                    if word[i] != char:
                        match = False
                        break
                elif word[i] in guessed_letters:
                    match = False
                    break
            
            if match:
                matching.append(word)
        
        return matching
    
    def _position_based_scoring(self, masked_word, emission_probs, remaining_letters):
        """Score letters based on position probabilities"""
        letter_scores = defaultdict(float)
        
        for pos, char in enumerate(masked_word):
            if char == '_':
                for letter in remaining_letters:
                    if letter in emission_probs[pos]:
                        letter_scores[letter] += emission_probs[pos][letter]
        
        return letter_scores
    
    def _frequency_based_scoring(self, words, masked_word, remaining_letters):
        """Score letters based on frequency in matching words"""
        letter_scores = defaultdict(int)
        
        for word in words:
            for pos, char in enumerate(masked_word):
                if char == '_':
                    letter = word[pos]
                    if letter in remaining_letters:
                        letter_scores[letter] += 1
        
        return letter_scores
    
    def _fallback_probabilities(self, guessed_letters, length):
        """Fallback to general letter frequencies"""
        if length in self.letter_frequencies:
            freq = self.letter_frequencies[length]
        else:
            all_freq = {}
            for freq_dict in self.letter_frequencies.values():
                for letter, prob in freq_dict.items():
                    all_freq[letter] = all_freq.get(letter, 0) + prob
            total = sum(all_freq.values())
            freq = {k: v / total for k, v in all_freq.items()}
        
        remaining = set('abcdefghijklmnopqrstuvwxyz') - guessed_letters
        remaining_freq = {letter: freq.get(letter, 0.01) for letter in remaining}
        total = sum(remaining_freq.values())
        
        return {letter: prob / total for letter, prob in remaining_freq.items()}
    
    def save(self, filename='hmm_model.pkl'):
        """Save trained model"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'letter_frequencies': self.letter_frequencies
            }, f)
    
    def load(self, filename='hmm_model.pkl'):
        """Load trained model"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.models = data['models']
            self.letter_frequencies = data['letter_frequencies']
