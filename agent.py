import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LinearQNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    
    def train_step(self, state, action, reward, next_state, done):
        # Convert inputs to tensors if they're not already
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        
        if len(state.shape) == 1:
            # Only one sample
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        
        # Compute Q values for current states
        pred = self.model(state)
        
        # Clone for target values
        target = pred.clone()
        
        # Update target Q values with Bellman equation
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][action[idx].item()] = Q_new
        
        # Zero gradients, compute loss, backpropagate and optimize
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=100_000)
        self.model = LinearQNet(4, 256, 2)  # input: state_dim=4, output: actions=2
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)
    
    def get_action(self, state, exploit_only=False):
    # Use exploit_only=True for evaluation (no randomness)
        if not exploit_only:
           self.epsilon = max(0.01, 0.995 ** self.n_games * 80)
           if random.random() < self.epsilon / 100.0:
              return random.randint(0, 1)
        state = torch.tensor(np.array(state), dtype=torch.float)
        prediction = self.model(state)
        return torch.argmax(prediction).item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        if len(self.memory) > 1000:
            mini_sample = random.sample(self.memory, 1000)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        
        # Convert to numpy arrays first for better performance
        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
        # NOTE: self.n_games += 1 should now be handled in train.py when episode ends

