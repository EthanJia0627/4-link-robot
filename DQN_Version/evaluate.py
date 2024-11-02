import mujoco
import os
from DQN import *
from simulator_passive import *
from reward import *
import numpy as np

import mujoco.viewer

current_file_path = os.path.abspath(__file__)
save_path = os.path.dirname(current_file_path) + '/'
render = True

state_size = 4 * 3 + 3 * 2 + 1 * 6 + 9 + 1
action_size = 27
epsilon = 0  # No exploration during evaluation
dict_path = save_path + 'DQN_Model_Stand(Front).pth'

Trainner = DQN_Trainner(state_size, action_size, 0, 0, dict_path, epsilon)
Trainner.init_trainner()
Cri = Critic()
m = mujoco.MjModel.from_xml_path('/home/shatteredxz/Documents/AI Lab/Mujoco/model/4-link robot/test.xml')
d = mujoco.MjData(m)
speed = 0.5

if render:
    viewer = mujoco.viewer.launch_passive(m, d)
    Env = Simulator(m, d, viewer)
else:
    Env = Simulator(m, d)

print("\n[Evaluation]")
print(f"Desired Speed: {speed}")

while True:
    state = Env.get_state()
    state.append(speed)
    action = Trainner.step(state, False)
    Env.step(action)
    if render:
        if not viewer.is_running():
            break
        viewer.sync()
        time.sleep(0.01)
    
