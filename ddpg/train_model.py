"""
环境：
https://gymnasium.farama.org/
"""

import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

print(info)

episode_over = False
while not episode_over:
    action = (
        env.action_space.sample()
    )  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated

env.close()
