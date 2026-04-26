import mujoco
import torch
import torch.nn as nn
from rsl_rl.runners import OnPolicyRunner
import numpy as np

class StandEnv:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path("robot_arm.xml")
        self.data = mujoco.MjData(self.model)
    
    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        obs = self.data.qpos[7:]  # углы суставов
        reward = 1.0 - np.sum(np.abs(self.data.qvel[6:]))  # бонус за стояние
        done = False
        return obs, reward, done, {}
    
    def reset(self):
        self.data.qpos[7:] = 0.0  # стоячая поза
        return self.data.qpos[7:]

# MuJoCo RL обучение
env = StandEnv()
runner = OnPolicyRunner(env, ... )  # RSL-RL
runner.learn()