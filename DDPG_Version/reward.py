import torch

class Critic():
    def __init__(self):
        ## self.history是一个FIFO，可以存储10000个state数据
        self.history = []
        self.max_history_size = 10000
        self.total_reward = 0
    def reward(self,state):
        self.new_state(state)
        alarm = self.check_alarm()
        reward = 0
        if alarm:
            reward = -1000
        else:
            reward = self.calculate_reward()
        return alarm,reward
        ...
    def reset(self):
        ## 清空self.history
        self.history = []
        self.total_reward = 0
        ... 
    def new_state(self,state):
        ## 更新self.history
        # 拆分state到各个部分
        state = torch.FloatTensor(state).cuda()
        pos_4_links = state[:12].view(4, 3) # 4*3
        pos_speed_3_joints = state[12:18].view(3,2) # 3*2
        pos_com = state[18:21] # 3
        vel_com = state[21:24] # 3
        contact_ground = state[24:33] # 9
        desired_speed = state[33] # 1

        state = {
            "pos_4_links": pos_4_links,
            "pos_speed_3_joints": pos_speed_3_joints,
            "pos_com": pos_com,
            "vel_com": vel_com,
            "contact_ground": contact_ground,
            "desired_speed": desired_speed
        }
        self.history.append(state)
        if len(self.history) > self.max_history_size:
            self.history.pop(0)

    def calculate_reward(self):
        ## 根据state和self.history计算reward
        reward = 0
        if len(self.history) <= 1000:
            mean_vel_com = torch.mean(torch.stack([state["vel_com"][0]for state in self.history]))
        else:
            mean_vel_com = torch.mean(torch.stack([state["vel_com"][0]for state in self.history[-1000:]]))
        desired_speed = self.history[-1]["desired_speed"]
        reward += 10*torch.exp(-20*torch.abs(mean_vel_com-desired_speed)).item()

        contact_reward_table = torch.FloatTensor([1,1,-5,-5,-5,-5,1,1,-10]).cuda()
        contact_reward = torch.dot(self.history[-1]["contact_ground"],contact_reward_table)
        reward += contact_reward.item()

        if (torch.abs(self.history[-1]["pos_speed_3_joints"][2])>6).any():
            reward -= 100
        

        self.total_reward += reward
        return reward
    def check_alarm(self):
        if self.history[-1]["contact_ground"][8] == 100:
            print("Alarm: Pushing Through Model.")
            return True
        if (abs(self.history[-1]["pos_4_links"][:,2])>6).any():
            print("Alarm: Z axis Locomotion Overrange.")
            return True
        # if (torch.abs(self.history[-1]["pos_speed_3_joints"][2])>5).any():
        #     print("Warning: Joint Overspeed.")
        #     return False
        return False