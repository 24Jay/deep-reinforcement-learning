import numpy as np
import random
import torch
import gymnasium as gym
import os
from ddpg_models import DDPGModel, ReplayBuffer
import matplotlib.pyplot as plt
import datetime


ENV_NAME = "Pendulum-v1"
ENV_NAME = "MountainCarContinuous-v0"

env = gym.make(ENV_NAME, render_mode="human")
ddpg = torch.load(f"ddpg_{ENV_NAME}.pth", weights_only=False)

# ddpg.eval()
state, _ = env.reset()

done = False
r = 0
i = 0
while not done:
    action = ddpg.get_action(state, eval=True)
    # print(action)
    next_state, reward, terminated, truncated, info = env.step(action)
    # done = terminated
    done = terminated or truncated
    state = next_state
    r += reward
    i += 1
    print(f"#{i*"#"}, {r=}, {reward=}")
