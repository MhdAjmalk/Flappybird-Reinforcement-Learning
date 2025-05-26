import torch
import os
import numpy as np
from agent import Agent
from env import FlappyBirdEnv
import time

def evaluate_single_episode(model_path, render=True, delay=0.05):
    """Evaluate the model for a single episode"""
    # Initialize the Flappy Bird Environment
    env = FlappyBirdEnv()
    
    # Initialize the Agent and load the trained model
    agent = Agent()
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return 0
    
    try:
        # Load the model with device compatibility
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent.model.load_state_dict(torch.load(model_path, map_location=device))
        agent.model.eval()  # Set the model to evaluation mode
        print(f"Successfully loaded model: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return 0

    # Reset the environment to get the initial state
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0

    print("Starting evaluation...")
    
    while not done:
        # Get action from the agent using the current state
        action = agent.get_action(state, exploit_only=True)

        # Take a step in the environment with the selected action
        next_state, reward, done = env.step(action)
        
        # Accumulate the total reward
        total_reward += reward
        steps += 1
        
        # Optionally render the game screen and add delay
        if render:
            env.render()
            if delay > 0:
                time.sleep(delay)

        # Update the state for the next step
        state = next_state

    return total_reward

def evaluate_multiple_episodes(model_path, num_episodes=5, render=True, delay=0.05):
    """Evaluate the model across multiple episodes"""
    print(f"Evaluating model across {num_episodes} episodes...")
    print("=" * 50)
    
    scores = []
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        score = evaluate_single_episode(model_path, render=render, delay=delay)
        scores.append(score)
        print(f"Episode {episode + 1} Score: {score}")
    
    # Calculate statistics
    avg_score = np.mean(scores)
    best_score = max(scores)
    worst_score = min(scores)
    std_score = np.std(scores)
    
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS:")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Best Score: {best_score}")
    print(f"Worst Score: {worst_score}")
    print(f"Standard Deviation: {std_score:.2f}")
    print(f"All Scores: {scores}")
    
    return scores

def evaluate(model_path="best_flapppppy_bird_model.pth", num_episodes=1, render=True, delay=0.05):
    """Main evaluation function"""
    print(f"Flappy Bird AI Evaluation")
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Render: {render}")
    print(f"Delay: {delay}s")
    print("-" * 30)
    
    if num_episodes == 1:
        score = evaluate_single_episode(model_path, render=render, delay=delay)
        print(f"\nFinal Score: {score}")
        return [score]
    else:
        return evaluate_multiple_episodes(model_path, num_episodes=num_episodes, render=render, delay=delay)

if __name__ == "__main__":
    # Single episode evaluation with visualization
    print("Single Episode Evaluation:")
    evaluate("best_flapppppy_bird_model.pth", num_episodes=1, render=True, delay=0.05)
    
    # Uncomment below for multiple episode evaluation without visualization (faster)
    # print("\n" + "="*60)
    # print("Multiple Episode Evaluation:")
    # evaluate("best_flapppppy_bird_model.pth", num_episodes=10, render=False, delay=0)
