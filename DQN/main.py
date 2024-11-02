import mujoco
import mujoco.viewer
import csv
import os
from DQN import *
from simulator_passive import *
from reward import *
current_file_path = os.path.abspath(__file__)
save_path = os.path.dirname(current_file_path)+'/'
render = True
# position of 4 links + position and speed of 3 joints + position and velocity of CoM + value standing for contact with ground + desired speed
state_size = 4*3 + 3*2 + 1*6 + 9 + 1
# action of 3 joints
action_size = 27
batch_size = 256
learning_rate = 0.000001
epsilon = 0.6
dict_path = None
dict_path = save_path+'DQN_Model_new.pth'
Init = True
com_pos_temp = None
activation = 3.14
Trainner = DQN_Trainner(state_size,action_size,batch_size,learning_rate,dict_path,epsilon)
Cri = Critic()
render = True
m = mujoco.MjModel.from_xml_path('/home/shatteredxz/Documents/AI Lab/Mujoco/model/4-link robot/test.xml')
# m.opt.timestep = 0.05
d = mujoco.MjData(m)
speed = 0.5


with mujoco.viewer.launch_passive(m, d) as viewer:
  if render:
    Env = Simulator(m,d,viewer)
  else:
    Env = Simulator(m,d)

  Trainner.init_trainner()
  for Episode in range(10000):
    speed = 4*np.random.rand()-2
    print("\n[Train]")
    print(f"Episode:{Episode+1},Desired Speed:{speed}")
    while Env.time() <=10:
      state = Env.get_state()
      state.append(speed)
      Alarm,reward = Cri.reward(state)
      Trainner.feedback(reward)
      action = Trainner.step(state)
      if Alarm:
        break
      Env.step(action)
      # Env.step()
      # print(reward,state)
    with open(save_path+'rewards.csv', 'a', newline='') as csvfile:
      csv_writer = csv.writer(csvfile)
      # 如果是新文件，写入表头
      if csvfile.tell() == 0:
          csv_writer.writerow(['Episode', 'Total Reward'])
      csv_writer.writerow([Episode + 1, Trainner.total_reward])
      csvfile.flush()  # 确保每次写入后数据都保存到文件
    Trainner.round_over()
    Env.rest()
    Cri.reset()
