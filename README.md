# Hangman AI: HMM + Reinforcement Learning

## Project Overview
An intelligent Hangman solver combining Hidden Markov Models (HMM) for pattern recognition with Reinforcement Learning (RL) for strategic decision-making.

## File Structure
```
project/
├── corpus.txt                 # 50,000 word training corpus
├── test_words.txt            # Test set for evaluation
├── hangman_hmm.py            # HMM implementation
├── hangman_env.py            # Game environment
├── hangman_agent.py          # RL agent
├── train_pipeline.py         # Complete training script
├── evaluate.py               # Evaluation on test set
├── demo.py                   # Interactive demo
└── requirements.txt          # Python dependencies
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements.txt
```
numpy>=1.21.0
matplotlib>=3.4.0
scipy>=1.7.0
```

## Quick Start

### 1. Train the Complete System
```bash
python train_pipeline.py
```

This will:
- Train the HMM on corpus.txt
- Train the RL agent for 10,000 episodes
- Save models (hmm_model.pkl, rl_agent.pkl)
- Generate training plots

### 2. Evaluate on Test Set
```bash
python evaluate.py
```

### 3. Play Interactive Demo
```bash
python demo.py
```
