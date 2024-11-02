import matplotlib.pyplot as plt
import csv
import numpy as np
import os
current_file_path = os.path.abspath(__file__)
save_path = os.path.dirname(current_file_path)+'/'
# 读取 CSV 文件中的数据
episodes = []
rewards = []

with open(save_path+'rewards.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)  # 跳过表头
    for row in csv_reader:
        episodes.append(int(row[0]))
        rewards.append(float(row[1]))

# 可视化 total reward
plt.plot(episodes, rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.show()
