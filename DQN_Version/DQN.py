import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

current_file_path = os.path.abspath(__file__)
save_path = os.path.dirname(current_file_path)+'/'

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
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


class DQN_Trainner():
    def __init__(self,input_size = 4 ,output_size = 2 ,batch_size = 64, learning_rate = 0.001,dict_path = None,epsilon = 1.0):
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = 0.99
        self.epsilon = epsilon
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.target_update = 5 
        self.policy_net = DQN(self.input_size, self.output_size).cuda()
        self.target_net = DQN(self.input_size, self.output_size).cuda()
        if dict_path:
            self.target_net.load_state_dict(torch.load(dict_path)) 
        self.policy_net.load_state_dict(self.target_net.state_dict())
        self.state = None
        self.next_state = None
        self.action = None
        self.reward = 0
        self.total_reward = 0
        self.round = 0
        self.total_reward_max = -100000
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(capacity=10000000)


    def train_model_from_replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = torch.FloatTensor(state).cuda()
        next_state = torch.FloatTensor(next_state).cuda()
        action = torch.LongTensor(action).cuda()
        reward = torch.FloatTensor(reward).cuda()
        done = torch.FloatTensor(done).cuda()
        q_values = self.policy_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_state).max(1)[0]
        expected_q_values = reward + self.gamma * next_q_values * (1 - done)
        loss = nn.MSELoss()(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()

    def init_trainner(self):
        self.state = None
        self.total_reward = 0
        self.next_state = None
        self.reward = 0
        
    def step(self,state,train = True):
        if train:
            self.next_state = state
            if self.state != None:
                self.replay_buffer.push(self.state, self.action, self.reward, self.next_state,False) 
                self.train_model_from_replay()
        if random.random() < self.epsilon:
            action = int(self.output_size*np.random.rand())
        else:
            q_values = self.policy_net(torch.FloatTensor(state).cuda())
            action = q_values.argmax().item()
        self.action = action
        self.state = state
        return action

    def feedback(self,reward):
        self.reward = reward
        self.total_reward += reward
        
    def round_over(self):
        self.epsilon = max(self.epsilon_min,self.epsilon * self.epsilon_decay)
        self.round += 1
        print(f"Round: {self.round}  Epsilon: {self.epsilon}")
        print(f"Total Reward: {self.total_reward}")
        if self.round == self.target_update:
            self.round = 0
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print("#####Target_Net Updated.#####")
        if self.total_reward >= self.total_reward_max:
            self.total_reward_max = self.total_reward
            self.save_model()
        self.init_trainner()

    def save_model(self,path = save_path, name = ''):
        torch.save(self.policy_net.state_dict(),path+'DQN_Model'+name + '.pth')

        
# # ��ʼѵ��
# num_episodes = 1000
# for episode in range(num_episodes):
#     state = env.reset()
#     if (len(state)==2):
#         state = state[0]
#     total_reward = 0
#     done = False
#     while not done:
#         if random.random() < epsilon:
#             action = env.action_space.sample()
#         else:
#             q_values = policy_net(torch.FloatTensor(state).cuda())
#             action = q_values.argmax().item()

#         next_state, reward, done, _ , _ = env.step(action)
#         total_reward += reward
#         if (len(state)==2):
#             state = state[0]
#         if (len(next_state)==2):
#             next_state = next_state[0]
#         replay_buffer.push(state, action, reward, next_state, done)
#         state = next_state

#         train_model()
#         if total_reward>=10000:
#            break

#     epsilon = max(epsilon_min, epsilon * epsilon_decay)

#     if episode % target_update == 0:
#         target_net.load_state_dict(policy_net.state_dict())

#     print(f"Episode {episode + 1}, Reward: {total_reward}")

# # ����ģ��
# torch.save(policy_net.state_dict(),'fishing.pth')
