import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class BandwidthAgent:
    def __init__(self, state_dim, action_dim, config):
        train_cfg = config['training']
        self.action_dim = action_dim
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), 
                                   lr=train_cfg['learning_rate'])
        self.memory = deque(maxlen=train_cfg['memory_capacity'])
        self.steps_done = 0
        self.eps_threshold = train_cfg['eps_start']
        self.eps_end = train_cfg['eps_end']
        self.eps_decay = train_cfg['eps_decay']
        self.gamma = train_cfg['gamma']
        self.batch_size = train_cfg['batch_size']
        self.tau = train_cfg['tau']
        self.losses = []
    
    def select_action(self, state):
        sample = random.random()
        self.eps_threshold = max(self.eps_end, self.eps_threshold * self.eps_decay)
        
        if sample > self.eps_threshold:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                action = torch.sigmoid(q_values).squeeze().numpy()
                return action
        else:
            action = np.random.rand(self.action_dim).clip(0.1, 0.9)
            action[0] = min(0.9, action[0] * 1.5)  # Bias toward critical
            action[1] = min(0.9, action[1] * 1.2)
            return action
    
    def store_transition(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        
        current_q = self.policy_net(states)
        with torch.no_grad():
            next_q = self.target_net(next_states)
            max_next_q, _ = next_q.max(1)
        
        target_q = rewards + self.gamma * max_next_q
        q_expected = current_q.gather(1, actions.argmax(dim=1, keepdim=True)).squeeze()
        
        loss = nn.MSELoss()(q_expected, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self._update_target_network()
        self.losses.append(loss.item())
        return loss.item()
    
    def _update_target_network(self):
        for target_param, policy_param in zip(self.target_net.parameters(), 
                                             self.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data
            )
    
    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
    
    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()