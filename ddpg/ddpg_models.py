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
        # self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_action)
        # 删除self.relu = nn.ReLU()，因为我们在forward中直接使用F.relu()

    def forward(self, state):
        # print(f"state dim must be 3: {state.shape=}")
        x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.tanh(x) * 2


class Critic(nn.Module):

    def __init__(self, n_state: int, n_action: int, n_hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_state + n_action, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, 1)

    def forward(self, state, action):
        # print(f"state dim must be 3: {state=}, {action.shape=}")

        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # print(f"==============={state.shape=}, {action.shape=}, {x.shape=}")

        return x


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

        # 高斯噪声，均值为0
        self.sigma = sigma

        self.device = "cpu"

        self.cnt = 0

    def get_action(self, state, eval=False):

        state = torch.FloatTensor(state).to(self.device)
        # print(f"state dim must be 3: {state.shape=}")

        a1 = self.actor(state)
        # print(f"============={a1=}")

        if eval:
            return a1.detach().numpy()
        action = a1 + torch.randn(a1.shape) * self.sigma
        # print(f"-------------{action.detach().numpy()=}")
        # print(f"============={a1.item()=}, {action.item()=}")

        return action.detach().numpy()

    def soft_update(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def update(self, transition_dict):
        state = torch.FloatTensor(transition_dict["states"]).to(self.device)
        action = torch.FloatTensor(transition_dict["actions"]).to(self.device)
        reward = (
            torch.FloatTensor(transition_dict["rewards"]).to(self.device).view(-1, 1)
        )
        next_state = torch.FloatTensor(transition_dict["next_states"]).to(self.device)
        done = torch.FloatTensor(transition_dict["dones"]).to(self.device).view(-1, 1)
        # print(f"check type: {state.shape=}, {action.shape=}, {reward.shape=}, {next_state.shape=}, {done.shape=}")
        if torch.sum(done) > 0:
            # print(f"==============================={torch.sum(done)=}")
            pass

        # update critic by MSE loss
        next_q_value = self.critic_target(next_state, self.actor_target(next_state))
        # print(f"check type: {next_state.shape=}, {action.shape=}")

        q_target = reward + self.gamma * next_q_value * (1 - done)
        # q_target = q_target.view(-1, 1)

        # print(f"shape: ")
        # print(f"check type: {next_q_value.shape=}, {q_target.shape=}, {reward.shape=}")

        critic_loss = torch.mean(F.mse_loss(self.critic(state, action), q_target))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # nn.utils.clip_grad_value_(self.critic.parameters(), clip_value=1)

        self.critic_optimizer.step()

        # update actor by policy gradient loss
        action_critics = self.critic(state, self.actor(state))
        action_loss = -action_critics.mean()
        # print(f"=====check type: {action_critics.shape=}, {action_loss=}")

        self.actor_optimizer.zero_grad()
        action_loss.backward()
        self.actor_optimizer.step()

        # soft update
        self.soft_update(self.critic, self.critic_target, self.tau)
        self.soft_update(self.actor, self.actor_target, self.tau)

        self.cnt += 1
        if self.cnt % 1000 == 0:
            print(
                f"{self.cnt=}, critic_loss: {critic_loss.item():.4f}, action_loss: {action_loss.item():.4f}, { self.tau:}"
            )
