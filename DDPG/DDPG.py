import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
current_file_path = os.path.abspath(__file__)
save_path = os.path.dirname(current_file_path)+'/'

class Actor(nn.Module):
    def __init__(self, input_size, output_size, activation):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Tanh(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Tanh()  # 使用Tanh来限制输出范围
        )
        self.scaler = torch.FloatTensor([activation]).cuda()
        self.initialize_weights()
    
    def initialize_weights(self):
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.fc(x)*self.scaler

class Critic(nn.Module):
    def __init__(self, input_size, action_size):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size + action_size, 128),
            nn.Tanh(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.initialize_weights()
    
    def initialize_weights(self):
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class DDPG_Trainer:
    def __init__(self, input_size=4, output_size=2, activation = 1,batch_size=64, learning_rate=0.0001, dict_path=None, epsilon=1.0):
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = 0.99
        self.epsilon = epsilon
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.target_update = 5
        self.activation = activation
        self.actor = Actor(self.input_size, self.output_size,activation).cuda()
        self.target_actor = Actor(self.input_size, self.output_size,activation).cuda()
        self.critic = Critic(self.input_size, self.output_size).cuda()
        self.target_critic = Critic(self.input_size, self.output_size).cuda()
        
        if dict_path:
            self.target_actor.load_state_dict(torch.load(dict_path[0]))
            self.target_critic.load_state_dict(torch.load(dict_path[1]))

        self.actor.load_state_dict(self.target_actor.state_dict())
        self.critic.load_state_dict(self.target_critic.state_dict())

        self.state = None
        self.next_state = None
        self.action = np.zeros(output_size)
        self.reward = 0
        self.total_reward = 0
        self.round = 0
        self.total_reward_max = -100000

        self.target_actor.eval()
        self.target_critic.eval()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(capacity=100000)

    def update_networks(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = torch.FloatTensor(state).cuda()
        next_state = torch.FloatTensor(next_state).cuda()
        action = torch.FloatTensor(action).cuda()
        reward = torch.FloatTensor(reward).cuda()
        done = torch.FloatTensor(done).cuda()

        # 更新Critic
        next_action = self.target_actor(next_state)
        target_q_value = self.target_critic(next_state, next_action)
        expected_q_value = reward.unsqueeze(1) + self.gamma * target_q_value * (1 - done.unsqueeze(1))
        q_value = self.critic(state, action)
        critic_loss = nn.MSELoss()(q_value, expected_q_value.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新Actor
        policy_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

    def init_trainner(self):
        self.state = None
        self.total_reward = 0
        self.next_state = None
        self.reward = 0
        self.action = np.zeros(self.output_size)

    def step(self, state, action_range=[]):
        self.next_state = state
        if self.state is not None:
            self.replay_buffer.push(self.state, self.action, self.reward, self.next_state, False)
            self.update_networks()

        if random.random() < self.epsilon:
            if len(action_range) == 2:
                action = (action_range[1] - action_range[0]) * np.random.rand(self.output_size) + action_range[0]
            else:
                action = np.clip(np.add(self.action, (action_range[3] - action_range[2]) * np.random.rand(self.output_size) + action_range[2]), action_range[0], action_range[1])
        else:
            action = (self.actor(torch.FloatTensor(state).cuda())).detach().cpu().numpy()
        self.action = action
        self.state = state
        return action

    def feedback(self, reward):
        self.reward = reward
        self.total_reward += reward

    def round_over(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.round += 1
        print(f"Round: {self.round}  Epsilon: {self.epsilon}")
        print(f"Total Reward: {self.total_reward}")
        if self.round == self.target_update:
            self.round = 0
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_critic.load_state_dict(self.critic.state_dict())
            print("Target_Net Updated.")
        if self.total_reward >= self.total_reward_max:
            self.total_reward_max = self.total_reward
            self.save_model()
        self.init_trainner()

    def save_model(self, path=save_path, name = ''):
        torch.save(self.actor.state_dict(),path + 'Actor_DDPG_Model' + name +'.pth')
        torch.save(self.critic.state_dict(), path + 'Critic_DDPG_Model' + name +'.pth')
