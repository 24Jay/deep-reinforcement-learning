import torch
import torch.nn as nn
from collections import deque


class Actor(nn.Module):

    def __init__(self, n_state: int, n_action: int, n_hidden: int = 64) -> None:
        super(Actor).__init__()


class Critic(nn.Module):

    def __init__(self, n_state: int, n_action: int, n_hidden: int = 64) -> None:
        super(Actor).__init__()


class ReplayBuffer:
    def __init__(self, capacity) -> None:
        self.capacity = capacity
        self.queue = deque()

    def __len__(self):
        return len(self.queue)
