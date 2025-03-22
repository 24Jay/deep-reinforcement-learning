import numpy as np
import random
import torch
import gymnasium as gym
import os
from ddpg_models import DDPGModel, ReplayBuffer
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("./refinforcement/ddpg")


ENV_NAME = "MountainCarContinuous-v0"
ENV_NAME = "Pendulum-v1"
EPOCH = 2000
MAX_STEP = 10

# env = gym.make(ENV_NAME, render_mode="human")
env = gym.make(ENV_NAME)


gamma = 0.99
tau = 0.01
buffer_size = 10000
min_buffer_size = 1000
batch_size = 65

LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
sigma = 0.01
print("action space:", env.action_space)
print("state space:", env.observation_space)

n_action = env.action_space.shape[0]
n_state = env.observation_space.shape[0]


random.seed(42)
np.random.seed(42)
# env.seed(0)
torch.manual_seed(42)

max_action = env.action_space.high[0]

print(f"max: {max_action}")

replay_buffer = ReplayBuffer(capacity=buffer_size)


ddpg = DDPGModel(
    n_state,
    n_action,
    n_hidden=64,
    sigma=sigma,
    lr_actor=LR_ACTOR,
    lr_critic=LR_CRITIC,
    gamma=gamma,
    tau=tau,
)


epsion = 0.05

returns = []

for epoch in range(EPOCH):

    done = False
    state, _ = env.reset(seed=42)
    episode_reward = 0
    i = 0
    while not done:
        action = ddpg.get_action(state, eval=True)
        if np.random.rand() < epsion:
            action = env.action_space.sample()
        else:
            action = action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state

        replay_buffer.push(state, action, reward, next_state, terminated)
        episode_reward += reward

        if len(replay_buffer) > min_buffer_size:
            batch_samples = replay_buffer.sample(batch_size)

            transition_dict = {
                "states": [x[0] for x in batch_samples],
                "actions": [x[1] for x in batch_samples],
                "rewards": [x[2] for x in batch_samples],
                "next_states": [x[3] for x in batch_samples],
                "dones": [x[4] for x in batch_samples],
            }
            ddpg.update(transition_dict)
        i += 1
    # print("*"*i, i, reward)

    returns.append(episode_reward)

    print(f"# {epoch}, 平均奖励：{episode_reward}")

    writer.add_scalar("reward", episode_reward, epoch)


# ddpg.save_model("ddpg_model.pth")

# plt.plot(returns)
# plt.show()
env.close()
writer.close()


env = gym.make(ENV_NAME, render_mode="human")

env.reset()

while not done:
    action = ddpg.get_action(state, eval=True)
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    state = next_state
