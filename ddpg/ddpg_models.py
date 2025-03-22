import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import collections
import random


class Actor(nn.Module):

    def __init__(self, n_state: int, n_action: int, n_hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_state, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_action)

    def forward(self, state):
        # 确保输入state的维度正确
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # 将输出限制在[-2, 2]范围内
        return torch.tanh(x) * 2


class Critic(nn.Module):

    def __init__(self, n_state: int, n_action: int, n_hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_state + n_action, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, 1)

    def forward(self, state, action):
        # 确保输入维度正确
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        x = torch.cat([state, action], dim=2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:

    def __init__(self, capacity: int) -> None:
        self.buffer = collections.deque(maxlen=capacity)

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))


class DDPGModel:

    def __init__(
        self,
        n_state: int,
        n_action: int,
        n_hidden: int = 64,
        lr_actor: float = 0.001,
        lr_critic: float = 0.001,
        gamma: float = 0.98,
        tau: float = 0.005,
        sigma: float = 0.1,
    ) -> None:
        self.actor = Actor(n_state, n_action, n_hidden)
        self.actor_target = Actor(n_state, n_action, n_hidden)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(n_state, n_action, n_hidden)
        self.critic_target = Critic(n_state, n_action, n_hidden)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = gamma
        self.tau = tau
        self.sigma = sigma
        self.device = "cpu"
        self.cnt = 0

    def get_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).view(1, -1)
        # with torch.no_grad():
        # action = self.actor(state)
        # if not eval:
        #     # 添加探索噪声
        #     noise = torch.randn_like(action) * self.sigma
        #     action = action + noise

        action = self.actor(state).item()

        if eval:
            return action
        else:
            # 给动作添加噪声，增加探索
            r = self.sigma * np.random.randn(1)

            # print(f"======{action=}, {r=}")
            return action + r

    def soft_update(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def update(self, transition_dict):
        # 转换数据为tensor
        # print(f"========={transition_dict["states"]=}")

        state = torch.tensor(
            np.array(transition_dict["states"]), dtype=torch.float32
        ).view(-1, 1, 3)

        # print(f"========={transition_dict["actions"]=}")
        # print(f"========={np.array(transition_dict["actions"]).shape=}")

        action = (
            torch.tensor(np.array(transition_dict["actions"]), dtype=torch.float32)
            .unsqueeze(0)
            .reshape(-1, 1, 1)
        )
        reward = (
            torch.FloatTensor(np.array(transition_dict["rewards"]))
            .unsqueeze(0)
            .reshape(-1, 1, 1)
        )

        next_state = torch.FloatTensor(np.array(transition_dict["next_states"])).view(
            -1, 1, 3
        )
        done = (
            torch.FloatTensor(np.array(transition_dict["dones"]))
            .to(self.device)
            .view(-1, 1, 1)
        )
        # print(f"check type: {state.shape=}, {action.shape=}, {reward.shape=}, {next_state.shape=}, {done.shape=}")
        if torch.sum(done) > 0:
            # print(f"==============================={torch.sum(done)=}")
            pass

        # 更新critic
        # with torch.no_grad():
        next_action = self.actor_target(next_state)
        next_q_value = self.critic_target(next_state, next_action)
        q_target = reward + self.gamma * next_q_value * (1 - done)

        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新actor
        policy_action = self.actor(state)
        action_loss = -self.critic(state, policy_action).mean()

        self.actor_optimizer.zero_grad()
        action_loss.backward()
        # 对actor网络的梯度进行裁剪，防止梯度爆炸
        # torch.nn.utils.clip_grad_norm_(self.actor.paramzseters(), max_norm=1.0)
        self.actor_optimizer.step()

        # 软更新目标网络
        self.soft_update(self.critic, self.critic_target, self.tau)
        self.soft_update(self.actor, self.actor_target, self.tau)

        self.cnt += 1
        if self.cnt % 1000 == 0:
            print(
                f"{self.cnt=}, critic_loss: {critic_loss.item():.4f}, action_loss: {action_loss.item():.4f}, {self.tau:}"
            )
