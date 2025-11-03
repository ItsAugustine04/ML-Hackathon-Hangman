# Reinforcement Learning Agent for Hangman
import numpy as np
import random
from collections import defaultdict

class HangmanRLAgent:
    """
    Q-Learning agent that uses HMM probabilities to play Hangman
    """
    
    def __init__(self, hmm_model, learning_rate=0.1, discount_factor=0.95, 
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
        self.hmm = hmm_model
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        self.training_stats = {
            'episodes': [],
            'rewards': [],
            'wins': [],
            'wrong_guesses': [],
        }
    
    def _get_state_key(self, state):
        """Convert state to hashable key for Q-table"""
        masked = state['masked_word']
        guessed = tuple(sorted(state['guessed_letters']))
        lives = state['lives_remaining']
        
        return (masked, guessed, lives)
    
    def choose_action(self, state, available_actions, training=True):
        """Choose action using epsilon-greedy policy with HMM guidance"""
        if not available_actions:
            return None
        
        hmm_probs = self.hmm.predict_letter_probabilities(
            state['masked_word'],
            state['guessed_letters']
        )
        
        if training and random.random() < self.epsilon:
            letters = [l for l in available_actions if l in hmm_probs]
            if not letters:
                return random.choice(available_actions)
            
            probs = [hmm_probs.get(l, 0.001) for l in letters]
            total = sum(probs)
            probs = [p / total for p in probs]
            
            return np.random.choice(letters, p=probs)
        else:
            state_key = self._get_state_key(state)
            
            best_score = -float('inf')
            best_action = None
            
            for action in available_actions:
                q_value = self.q_table[state_key][action]
                hmm_score = hmm_probs.get(action, 0.001) * 10
                combined_score = q_value + hmm_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_action = action
            
            return best_action if best_action else random.choice(available_actions)
    
    def update_q_value(self, state, action, reward, next_state, done):
        """Update Q-value using Q-learning update rule"""
        state_key = self._get_state_key(state)
        
        current_q = self.q_table[state_key][action]
        
        if done:
            max_next_q = 0
        else:
            next_state_key = self._get_state_key(next_state)
            next_q_values = self.q_table[next_state_key]
            max_next_q = max(next_q_values.values()) if next_q_values else 0
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state_key][action] = new_q
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def train(self, env, num_episodes=10000, verbose=True):
        """Train agent through self-play"""
        print(f"Starting training for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            wrong_guesses = 0
            
            while not state['game_over']:
                available = env.get_available_actions()
                action = self.choose_action(state, available, training=True)
                
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                
                if not info['correct'] and not info['repeated']:
                    wrong_guesses += 1
                
                self.update_q_value(state, action, reward, next_state, done)
                
                state = next_state
            
            self.decay_epsilon()
            
            self.training_stats['episodes'].append(episode)
            self.training_stats['rewards'].append(total_reward)
            self.training_stats['wins'].append(1 if state['won'] else 0)
            self.training_stats['wrong_guesses'].append(wrong_guesses)
            
            if verbose and (episode + 1) % 1000 == 0:
                recent_wins = sum(self.training_stats['wins'][-1000:])
                recent_avg_wrong = np.mean(self.training_stats['wrong_guesses'][-1000:])
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"  Win rate (last 1000): {recent_wins/10:.1f}%")
                print(f"  Avg wrong guesses: {recent_avg_wrong:.2f}")
                print(f"  Epsilon: {self.epsilon:.4f}")
        
        print("Training complete!")
    
    def evaluate(self, env, num_games=1000, verbose=True):
        """Evaluate agent performance"""
        wins = 0
        total_wrong = 0
        total_repeated = 0
        
        for game in range(num_games):
            state = env.reset()
            wrong_guesses = 0
            repeated_guesses = 0
            
            while not state['game_over']:
                available = env.get_available_actions()
                action = self.choose_action(state, available, training=False)
                
                next_state, reward, done, info = env.step(action)
                
                if info['repeated']:
                    repeated_guesses += 1
                elif not info['correct']:
                    wrong_guesses += 1
                
                state = next_state
            
            if state['won']:
                wins += 1
            total_wrong += wrong_guesses
            total_repeated += repeated_guesses
            
            if verbose and (game + 1) % 200 == 0:
                print(f"Evaluated {game + 1}/{num_games} games...")
        
        success_rate = wins / num_games
        final_score = (success_rate * num_games) - (total_wrong * 5) - (total_repeated * 2)
        
        results = {
            'success_rate': success_rate,
            'wins': wins,
            'total_games': num_games,
            'total_wrong_guesses': total_wrong,
            'total_repeated_guesses': total_repeated,
            'final_score': final_score
        }
        
        if verbose:
            print("=" * 50)
            print("EVALUATION RESULTS")
            print("=" * 50)
            print(f"Success Rate: {success_rate*100:.2f}%")
            print(f"Final Score: {final_score:.2f}")
            print("=" * 50)
        
        return results
