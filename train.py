import torch
import numpy as np
from agent import Agent
from env import FlappyBirdEnv
import time
from tqdm import trange  # Progress bar

def train():
    # Initialize environment and agent
    env = FlappyBirdEnv()
    agent = Agent()

    # Handle GPU (if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.model.to(device)

    total_score = 0
    best_score = 0
    episodes = 100  # Number of training episodes

    # Track training time
    start_time = time.time()

    for episode in trange(episodes, desc="Training Progress"):
        state = env.reset()
        done = False
        score = 0

        while not done:
            # Choose action based on state
            action = agent.get_action(state)

            # Take action and get new state, reward, and done flag
            next_state, reward, done = env.step(action)

            # Remember the experience for training
            agent.remember(state, action, reward, next_state, done)

            # Train on short memory
            agent.train_short_memory(state, action, reward, next_state, done)

            # Move to the next state
            state = next_state

            # Add the reward to the score
            score += reward

            if done:
                # Long-term memory training
                agent.train_long_memory()

                # Update total score
                total_score += score
                avg_score = total_score / (episode + 1)

                print(f"Episode {episode + 1}/{episodes} | Score: {score} | Avg Score: {avg_score:.2f} | Total Score: {total_score}")

                # Save best model
                if score > best_score:
                    best_score = score
                    torch.save(agent.model.state_dict(), "best_flapppppy_bird_model.pth")

        # Save model periodically (every 50 episodes)
        if (episode + 1) % 50 == 0:
            torch.save(agent.model.state_dict(), f"flapppppy_bird_model_episode_{episode + 1}.pth")

    # Total training time
    end_time = time.time()
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    train()

