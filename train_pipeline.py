# Complete Training Pipeline
import matplotlib.pyplot as plt

def plot_training_results(agent):
    """Plot training statistics"""
    stats = agent.training_stats
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    window = 100
    
    # Plot 1: Rewards
    ax = axes[0, 0]
    rewards = stats['rewards']
    if len(rewards) >= window:
        moving_avg = [sum(rewards[i:i+window])/window 
                     for i in range(len(rewards)-window)]
        ax.plot(moving_avg)
    ax.set_title('Average Reward (100-episode window)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward')
    ax.grid(True)
    
    # Plot 2: Win rate
    ax = axes[0, 1]
    wins = stats['wins']
    if len(wins) >= window:
        win_rate = [sum(wins[i:i+window])/window * 100 
                   for i in range(len(wins)-window)]
        ax.plot(win_rate)
    ax.set_title('Win Rate (100-episode window)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Win Rate (%)')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300)
    plt.show()
