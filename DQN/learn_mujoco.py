import mujoco
import time
import itertools
import numpy as np
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2 as cv

# More legible printing from numpy.
# np.set_printoptions(precision=3, suppress=True, linewidth=100)

# xml = """
# <mujoco>
#   <compiler autolimits="true"/>

#   <option integrator="implicitfast"/>

#   <worldbody>
#     <geom type="plane" size="1 1 .01"/>
#     <light pos="0 0 2"/>
#     <body pos="0 0 .3">
#       <joint name="hinge" damping=".01" actuatorfrcrange="-.4 .4"/>
#       <geom type="capsule" size=".01" fromto="0 0 0 .2 0 0"/>
#       <geom size=".03" pos=".2 0 0"/>
#     </body>
#   </worldbody>

#   <actuator>
#     <motor name="motor" joint="hinge" ctrlrange="-1 1"/>
#     <damper name="damper" joint="hinge" kv="10" ctrlrange="0 1"/>
#   </actuator>

#   <sensor>
#     <actuatorfrc name="motor" actuator="motor"/>
#     <actuatorfrc name="damper" actuator="damper"/>
#     <jointactuatorfrc name="hinge" joint="hinge"/>
#   </sensor>
# </mujoco>
# """

# model = mujoco.MjModel.from_xml_string(xml)
model = mujoco.MjModel.from_xml_path('/home/shatteredxz/Documents/AI Lab/Mujoco/model/4-link robot/test.xml')
data = mujoco.MjData(model)
# model.opt.gravity = (0, 0, 10)
duration = 3.8  # (seconds)
framerate = 60  # (Hz)

# Simulate and display video.
frames = []
mujoco.mj_resetData(model, data)  # Reset state and time.
with mujoco.Renderer(model) as renderer:
    while data.time < duration:
        mujoco.mj_step(model, data)
        if len(frames) < data.time * framerate:
            renderer.update_scene(data)
            pixels = renderer.render()
        #   frames.append(cv.cvtColor(pixels,cv.COLOR_RGB2BGR))
            frames.append(pixels)

print('Total number of DoFs in the model:', model.nv)
print('Generalized positions:', data.qpos)
print('Generalized velocities:', data.qvel)

for frame in frames:
    cv.imshow("Render",cv.cvtColor(frame,cv.COLOR_RGB2BGR))
    cv.waitKey(1000//framerate)



