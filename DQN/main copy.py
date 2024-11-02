import mujoco
import mujoco.viewer
import numpy as np
import time
import torch


render = True
# position of 4 links + position and speed of 3 joints + position and velocity of CoM + Boolean value standing for contact with ground + desired speed
state_size = 4*3 + 3*2 + 1*6 + 8 + 1
# action of 3 joints
action_size = 3
batch_size = 64
learning_rate = 0.001
epsilon = 1.0
dict_path = None
Init = True
com_pos_temp = None




def key_callback(keycode):
  if chr(keycode) == ' ':
    global paused
    paused = not paused

def get_com(m,d):
  com = [0,0,0]
  total_mass = 9
  for i in range(2,6):
    com += m.body_mass[i]*d.xipos[i]
  com = com/total_mass  
  return com

def get_contact(m,d):
  contact = np.zeros(8)
  idx = [1,2,3,5,7,9,10,11]
  for pair in d.contact.geom:
    if pair[0]==0:
      contact[np.where(idx==pair[1])] = 1
  return contact

def get_state(m,d):
  link_pose = torch.FloatTensor(d.xpos[2:6]).view(-1).cuda()
  joint_state = torch.FloatTensor([d.qpos[7:10],d.qvel[6:9]]).view(-1).cuda()
  com_pos = get_com(m,d)
  global com_pos_temp,Init
  if Init:
    com_vel = [0,0,0]
    Init = False
  else:
    com_vel = com_pos-com_pos_temp
  com_state = torch.FloatTensor([com_pos,com_vel]).view(-1).cuda()
  com_pos_temp = com_pos
  contact = torch.FloatTensor(get_contact(m,d)).cuda()
  state = torch.cat([link_pose,joint_state,com_state,contact])
  return state

class Reward():
  def __init__(self):
    self.buffer = []
  def get_reward(state):
    ...

m = mujoco.MjModel.from_xml_path('/home/shatteredxz/Documents/AI Lab/Mujoco/model/4-link robot/test.xml')
d = mujoco.MjData(m)
contact_min = 1
if render:
  paused = False
  time_pre = time.time()
  runtime_pre = 0.
  with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
    while viewer.is_running():
      time_now = time.time()
      runtime = d.time
      time_gap = time_now-time_pre
      runtime_gap = runtime-runtime_pre
      if not paused :
        if runtime_gap<time_gap:
          time_pre = time_now
          runtime_pre = runtime
          # d.ctrl = np.random.rand(3)
          mujoco.mj_step(m, d)
          viewer.sync()
          if len(d.contact.dist):
            dist = np.min(d.contact.dist)
            if dist<contact_min:
              contact_min = dist
              print(dist)
      else:  
        time_pre = time.time()
        runtime_pre = runtime      



