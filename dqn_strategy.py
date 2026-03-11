"""
DQN-based strategy agent for high-level negotiation intent selection.
Implements Double DQN with prioritized experience replay.
"""
from __future__ import annotations

import random
from collections import deque, namedtuple
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

INTENTS = [
    "open_anchor",      # Set aggressive anchor
    "concede_small",    # Small concession to build rapport
    "hold_firm",        # No movement, signal strength
    "explore_zopa",     # Probe counterpart's limits
    "make_final",       # Signal best-and-final offer
    "accept",           # Accept current offer
    "reject",           # Reject and walk away
]


# ─── Network ──────────────────────────────────────────────────────────────────

class DQNNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        # Dueling architecture: value + advantage streams
        self.value_head = nn.Linear(hidden, 1)
        self.advantage_head = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.net(x)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))


# ─── Replay Buffer ────────────────────────────────────────────────────────────

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = 50_000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: deque[Transition] = deque(maxlen=capacity)
        self.priorities: deque[float] = deque(maxlen=capacity)

    def push(self, *args) -> None:
        max_priority = max(self.priorities, default=1.0)
        self.buffer.append(Transition(*args))
        self.priorities.append(max_priority)

    def sample(self, batch_size: int, beta: float = 0.4) -> tuple:
        priorities = np.array(self.priorities, dtype=np.float32) ** self.alpha
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        batch = [self.buffer[i] for i in indices]
        return batch, indices, torch.FloatTensor(weights)

    def update_priorities(self, indices: list[int], priorities: np.ndarray) -> None:
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = float(priority) + 1e-6

    def __len__(self) -> int:
        return len(self.buffer)


# ─── Agent ────────────────────────────────────────────────────────────────────

class DQNStrategyAgent:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.obs_dim = cfg.get("obs_dim", 5)
        self.n_actions = len(INTENTS)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQNNetwork(self.obs_dim, self.n_actions).to(self.device)
        self.target_net = DQNNetwork(self.obs_dim, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=cfg.get("lr", 1e-4))
        self.replay = PrioritizedReplayBuffer(cfg.get("buffer_capacity", 50_000))

        self.epsilon = cfg.get("epsilon_start", 1.0)
        self.epsilon_end = cfg.get("epsilon_end", 0.05)
        self.epsilon_decay = cfg.get("epsilon_decay", 0.995)
        self.gamma = cfg.get("gamma", 0.99)
        self.batch_size = cfg.get("batch_size", 128)
        self.target_update_freq = cfg.get("target_update_freq", 500)
        self.steps = 0

    async def select_intent(self, obs: dict) -> str:
        state_tensor = self._obs_to_tensor(obs)
        if random.random() < self.epsilon:
            action_idx = random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action_idx = q_values.argmax().item()
        return INTENTS[action_idx]

    def store_transition(self, obs: dict, intent: str, reward: float, next_obs: dict, done: bool) -> None:
        action_idx = INTENTS.index(intent)
        self.replay.push(
            self._obs_to_tensor(obs).cpu().numpy(),
            action_idx,
            reward,
            self._obs_to_tensor(next_obs).cpu().numpy(),
            done,
        )

    def train_step(self) -> float | None:
        if len(self.replay) < self.batch_size:
            return None

        batch, indices, weights = self.replay.sample(self.batch_size)
        weights = weights.to(self.device)

        states = torch.FloatTensor(np.array([t.state for t in batch])).to(self.device)
        actions = torch.LongTensor([t.action for t in batch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(self.device)
        dones = torch.FloatTensor([t.done for t in batch]).to(self.device)

        current_q = self.policy_net(states).gather(1, actions).squeeze()

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze()
            target_q = rewards + self.gamma * next_q * (1 - dones)

        td_errors = (current_q - target_q).abs().detach().cpu().numpy()
        self.replay.update_priorities(indices, td_errors)

        loss = (weights * nn.functional.smooth_l1_loss(current_q, target_q, reduction="none")).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.steps += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def _obs_to_tensor(self, obs: dict) -> torch.Tensor:
        vec = np.array([
            obs.get("phase", 0),
            obs.get("current_offer", 0.0),
            obs.get("counterpart_offer", 0.0),
            obs.get("round", 0) / 20.0,
            obs.get("zopa", 0.0),
        ], dtype=np.float32)
        return torch.FloatTensor(vec).unsqueeze(0).to(self.device)

    def save(self, path: str) -> None:
        torch.save({"policy": self.policy_net.state_dict(), "target": self.target_net.state_dict(), "steps": self.steps}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy"])
        self.target_net.load_state_dict(ckpt["target"])
        self.steps = ckpt.get("steps", 0)
