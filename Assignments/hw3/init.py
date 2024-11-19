from train import CarRacingNStepSarsa

def train_car_racing():
    """Train the agent on the Car Racing environment"""
    agent = CarRacingNStepSarsa(epsilon=0.1)  # 10% exploration rate
    num_episodes = 1000
    running_reward = None
    
    for episode in range(num_episodes):
        rewards = agent.run_episode()
        total_reward = sum(rewards)
        
        # Calculate running reward
        if running_reward is None:
            running_reward = total_reward
        else:
            running_reward = 0.05 * total_reward + 0.95 * running_reward
            
        print(f"Episode {episode + 1}, Total Reward: {total_reward:.2f}, Running Reward: {running_reward:.2f}")

if __name__ == "__main__":
    train_car_racing()
