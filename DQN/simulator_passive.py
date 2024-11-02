import mujoco
import mujoco.viewer
import numpy as np
import time
import torch




class Simulator():
  def __init__(self,m,d,viewer = None):
    self.m = m
    self.d = d
    self.Init = True
    self.com_pos_temp = None
    if viewer:
        self.render = True
        self.viewer = viewer
    else:
      self.render = False
    self.time_pre = time.time()
    self.runtime_pre = 0.
  def get_com(self):
    com = [0,0,0]
    total_mass = 9
    for i in range(2,6):
      com += self.m.body_mass[i]*self.d.xipos[i]
    com = com/total_mass  
    return com

  def get_contact(self):
    contact = np.zeros(9)
    idx = [1,2,3,5,7,9,10,11]
    for pair in self.d.contact.geom:
      if pair[0]==0:
        contact[np.where(idx==pair[1])] = 1
      else:
        contact[8] = 1
        if (self.d.contact.dist<-0.1).any():
          contact[8] = 100
    return contact

  def get_state(self):
    link_pose = torch.FloatTensor([self.d.xpos[i]-self.d.xpos[2] if i!=2 else self.d.xpos[2] for i in range(2,6)]).view(-1)
    joint_state = torch.FloatTensor([self.d.qpos[7:10],self.d.qvel[6:9]]).view(-1)
    com_pos = self.get_com()
    if self.Init:
      com_vel = [0,0,0]
      self.Init = False
    else:
      com_vel = com_pos-self.com_pos_temp
    com_state = torch.FloatTensor([com_pos,com_vel]).view(-1)
    self.com_pos_temp = com_pos
    contact = torch.FloatTensor(self.get_contact())
    state = torch.cat([link_pose,joint_state,com_state,contact]).tolist()
    return state

  def step(self,action = None):
    if self.d.time == 0:
      self.Init =True
    if self.render:
      if not self.viewer.is_running():
        return False
      while True:
        time_now = time.time()
        runtime = self.d.time
        time_gap = time_now-self.time_pre
        runtime_gap = runtime-self.runtime_pre
        # if runtime_gap<time_gap:
        if True:
          break
      self.time_pre = time_now
      self.runtime_pre = runtime
      if action:
        ctrl = np.clip(self.d.ctrl + self.action2ctrl(action,0.1),-3.14,3.14)
        self.d.ctrl = ctrl
      mujoco.mj_step(self.m,self.d,5)
      self.viewer.sync()
    else:
      mujoco.mj_step(self.m, self.d) 
    return True
  
  def rest(self):
    mujoco.mj_resetData(self.m,self.d)
    mujoco.mj_resetDataKeyframe(self.m,self.d,0)
    self.Init = True
  
  def time(self):
    return self.d.time

  def action2ctrl(self,action,step):
    ctrl = []
    for _ in range(3):
        ctrl.append(step*((action % 3)-1))
        action //=3
    return ctrl
