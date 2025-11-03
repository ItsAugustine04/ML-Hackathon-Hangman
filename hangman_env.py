# Hangman Game Environment
import random

class HangmanEnvironment:
    """
    Hangman game environment for RL agent training
    """
    
    def __init__(self, word_list_file='corpus.txt', max_wrong=6):
        self.max_wrong = max_wrong
        
        with open(word_list_file, 'r') as f:
            self.word_list = [line.strip().lower() for line in f 
                             if line.strip() and line.strip().isalpha()]
        
        print(f"Loaded {len(self.word_list)} words")
        
        self.target_word = None
        self.masked_word = None
        self.guessed_letters = set()
        self.wrong_guesses = 0
        self.game_over = False
        self.won = False
        
    def reset(self, word=None):
        """Reset environment for new game"""
        if word is None:
            self.target_word = random.choice(self.word_list)
        else:
            self.target_word = word.lower()
        
        self.masked_word = '_' * len(self.target_word)
        self.guessed_letters = set()
        self.wrong_guesses = 0
        self.game_over = False
        self.won = False
        
        return self._get_state()
    
    def step(self, letter):
        """Take action (guess a letter)"""
        letter = letter.lower()
        
        repeated = letter in self.guessed_letters
        self.guessed_letters.add(letter)
        
        reward = 0
        info = {'repeated': repeated, 'correct': False}
        
        if repeated:
            reward = -2
            info['msg'] = f"Already guessed '{letter}'"
        elif letter in self.target_word:
            info['correct'] = True
            
            new_masked = list(self.masked_word)
            occurrences = 0
            for i, char in enumerate(self.target_word):
                if char == letter:
                    new_masked[i] = letter
                    occurrences += 1
            self.masked_word = ''.join(new_masked)
            
            reward = 10 * occurrences
            info['msg'] = f"Correct! '{letter}' appears {occurrences} time(s)"
            
            if '_' not in self.masked_word:
                self.won = True
                self.game_over = True
                reward += 100
                info['msg'] = f"Won! Word was '{self.target_word}'"
        else:
            self.wrong_guesses += 1
            reward = -5
            info['msg'] = f"Wrong! '{letter}' not in word"
            
            if self.wrong_guesses >= self.max_wrong:
                self.game_over = True
                reward -= 50
                info['msg'] = f"Lost! Word was '{self.target_word}'"
        
        state = self._get_state()
        
        return state, reward, self.game_over, info
    
    def _get_state(self):
        """Get current state representation"""
        return {
            'masked_word': self.masked_word,
            'guessed_letters': self.guessed_letters.copy(),
            'wrong_guesses': self.wrong_guesses,
            'lives_remaining': self.max_wrong - self.wrong_guesses,
            'game_over': self.game_over,
            'won': self.won,
            'target_word': self.target_word
        }
    
    def get_available_actions(self):
        """Get list of letters that haven't been guessed"""
        all_letters = set('abcdefghijklmnopqrstuvwxyz')
        return list(all_letters - self.guessed_letters)
