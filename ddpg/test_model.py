import numpy as np
import random
import torch
import gymnasium as gym
import os
from ddpg_models import DDPGModel, ReplayBuffer
import matplotlib.pyplot as plt
import datetime


ENV_NAME = "Pendulum-v1"
EPOCH = 1000

env = gym.make(ENV_NAME, render_mode="human")


ddpg = torch.load("ddpg_pendulum.pth", weights_only=False)

# ddpg.eval()
state, _ = env.reset()

done = False

while not done:
    action = ddpg.get_action(state, eval=True)
    print(action)
    next_state, reward, terminated, truncated, info = env.step(action)
    # done = terminated
    done = terminated or truncated
    state = next_state
