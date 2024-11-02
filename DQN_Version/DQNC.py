import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# ����������ģ��


# ��������
class DQN(nn.Module):
    def __init__(self, input_size, output_size,activation = None):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        self.activation = activation
        if activation:
            self.out = nn.Tanh()
        self.initialize_weights()
    
    def initialize_weights(self):
        # 对每一层进行初始化
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                # Xavier 初始化
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
                
                # 或者使用 Kaiming 初始化
                # nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                # nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.fc(x)
        if self.activation:
          x = self.activation*self.out(x) 
        return x

# ���徭��طŻ���
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
    def __init__(self,input_size = 4 ,output_size = 2 ,activation_range = None,batch_size = 64, learning_rate = 0.001,dict_path = None,epsilon = 1.0):
        # ���ò���
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = 0.99
        self.epsilon = epsilon
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01
        self.target_update = 5  # Ŀ���������Ƶ��
        self.policy_net = DQN(self.input_size, self.output_size,activation= activation_range).cuda()
        self.target_net = DQN(self.input_size, self.output_size,activation= activation_range).cuda()
        self.Q_policy_net = DQN(self.input_size + self.output_size,1).cuda()
        self.Q_target_net = DQN(self.input_size + self.output_size,1).cuda()
        if dict_path:
            self.target_net.load_state_dict(torch.load(dict_path[0])) 
            self.Q_target_net.load_state_dict(torch.load(dict_path[1])) 
        self.policy_net.load_state_dict(self.target_net.state_dict())
        self.Q_policy_net.load_state_dict(self.Q_target_net.state_dict())
        self.state = None
        self.next_state = None
        self.action = np.zeros(output_size)
        self.reward = 0
        self.total_reward = 0
        self.round = 0
        self.total_reward_max = -100000
        # ��ʼ����������Ż���
        self.target_net.eval()
        self.Q_target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.Q_optimizer = optim.Adam(self.Q_policy_net.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(capacity=100000)

    # ѵ��ģ��
    def train_model_from_replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = torch.FloatTensor(state).cuda()
        next_state = torch.FloatTensor(next_state).cuda()
        action = torch.LongTensor(action).cuda()
        reward = torch.FloatTensor(reward).cuda()
        done = torch.FloatTensor(done).cuda()
        # update action policy network
        action_e = self.policy_net(state)
        q_value_predict = self.Q_policy_net(torch.cat([state,action_e],dim = 1))
        loss = -q_value_predict.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # update Q policy network
        next_action_predict = self.target_net(next_state)
        next_q_value_predict = self.Q_target_net(torch.cat([next_state,next_action_predict],dim = 1))
        q_value = self.Q_policy_net(torch.cat([state,action],dim = 1))
        expected_q_values = reward.unsqueeze(1) + self.gamma * next_q_value_predict * (1 - done.unsqueeze(1))
        loss = nn.MSELoss()(q_value, expected_q_values.detach())
        self.Q_optimizer.zero_grad()
        loss.backward()
        self.Q_optimizer.step()

    def init_trainner(self):
        self.state = None
        self.total_reward = 0
        self.next_state = None
        self.reward = 0
        self.action = np.zeros(self.output_size)
    def step(self,state,range = []):
        self.next_state = state
        if self.state != None:
            self.replay_buffer.push(self.state, self.action, self.reward, self.next_state,False) 
            self.train_model_from_replay()
        if random.random() < self.epsilon:
            if len(range)==2:
                if range:
                    action = (range[1]-range[0])*np.random.rand(self.output_size)+range[0]
                else:
                    action = np.random.rand(self.output_size)
            else:
                action =np.clip(np.add(self.action,(range[3]-range[2])*np.random.rand(self.output_size)+range[2]),
                                range[0],range[1]) 
        else:
            action = self.policy_net(torch.FloatTensor(state).cuda()).detach().cpu().numpy()
        self.action = action
        self.state = state
        return action

    def feedback(self,reward):
        self.reward = reward
        self.total_reward += reward
        
    def round_over(self):
        self.epsilon = max(self.epsilon_min,self.epsilon * self.epsilon_decay)
        print(f"epsilon:{self.epsilon}")
        self.round += 1
        print(f"round:{self.round}")
        print(f"Total Reward:{self.total_reward}")
        if self.round == self.target_update:
            self.round = 0
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.Q_target_net.load_state_dict(self.Q_policy_net.state_dict())
            print("Target_Net Updated.")
        if self.total_reward >= self.total_reward_max:
            self.total_reward_max = self.total_reward
            self.save_model()
        self.init_trainner()
        
    def save_model(self,path = 'Train_Model.pth'):
        torch.save(self.policy_net.state_dict(),'Action_'+path)
        torch.save(self.Q_policy_net.state_dict(),'Q_'+path)

        
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
