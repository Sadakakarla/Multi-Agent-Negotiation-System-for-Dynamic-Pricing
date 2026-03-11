"""
PPO self-play training for the dialogue agent.
Agents play against past versions of themselves to improve robustness.
"""
from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


@dataclass
class PPOBatch:
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    dones: torch.Tensor
    advantages: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    returns: torch.Tensor = field(default_factory=lambda: torch.tensor([]))


class PolicyValueNetwork(nn.Module):
    """Shared backbone with separate policy and value heads."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 512):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    def act(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value


class SelfPlayPPOTrainer:
    """
    PPO self-play: current policy plays against a frozen snapshot
    of itself from N episodes ago, then snapshots are updated periodically.
    """

    SNAPSHOT_INTERVAL = 200  # episodes between opponent snapshots
    OPPONENT_POOL_SIZE = 5

    def __init__(self, cfg: dict):
        self.cfg = cfg
        obs_dim = cfg.get("obs_dim", 32)
        n_actions = cfg.get("n_actions", 7)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = PolicyValueNetwork(obs_dim, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.get("lr", 3e-4), eps=1e-5)
        self.opponent_pool: deque[PolicyValueNetwork] = deque(maxlen=self.OPPONENT_POOL_SIZE)
        self._snapshot_policy()  # Seed pool with initial policy

        # PPO hyperparameters
        self.clip_eps = cfg.get("clip_eps", 0.2)
        self.vf_coef = cfg.get("vf_coef", 0.5)
        self.ent_coef = cfg.get("ent_coef", 0.01)
        self.n_epochs = cfg.get("n_epochs", 4)
        self.gae_lambda = cfg.get("gae_lambda", 0.95)
        self.gamma = cfg.get("gamma", 0.99)
        self.max_grad_norm = cfg.get("max_grad_norm", 0.5)
        self.episodes = 0

    def select_opponent(self) -> PolicyValueNetwork:
        """Sample uniformly from opponent pool for diversity."""
        return random.choice(list(self.opponent_pool))

    def _snapshot_policy(self) -> None:
        snapshot = copy.deepcopy(self.policy)
        snapshot.eval()
        for p in snapshot.parameters():
            p.requires_grad_(False)
        self.opponent_pool.append(snapshot)

    def compute_gae(self, batch: PPOBatch) -> PPOBatch:
        """Generalized Advantage Estimation."""
        advantages = torch.zeros_like(batch.rewards)
        gae = 0.0
        values_np = batch.values.cpu().numpy()
        rewards_np = batch.rewards.cpu().numpy()
        dones_np = batch.dones.cpu().numpy()

        for t in reversed(range(len(rewards_np))):
            next_val = values_np[t + 1] if t < len(values_np) - 1 else 0.0
            delta = rewards_np[t] + self.gamma * next_val * (1 - dones_np[t]) - values_np[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones_np[t]) * gae
            advantages[t] = gae

        returns = advantages + batch.values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        batch.advantages = advantages.to(self.device)
        batch.returns = returns.to(self.device)
        return batch

    def update(self, batch: PPOBatch) -> dict[str, float]:
        batch = self.compute_gae(batch)
        stats: dict[str, list[float]] = {"policy_loss": [], "value_loss": [], "entropy": [], "total_loss": []}

        for _ in range(self.n_epochs):
            logits, values = self.policy(batch.states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(batch.actions)
            entropy = dist.entropy().mean()

            ratio = (new_log_probs - batch.log_probs).exp()
            clipped_ratio = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps)
            policy_loss = -torch.min(ratio * batch.advantages, clipped_ratio * batch.advantages).mean()
            value_loss = nn.functional.mse_loss(values, batch.returns)
            total_loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy"].append(entropy.item())
            stats["total_loss"].append(total_loss.item())

        self.episodes += 1
        if self.episodes % self.SNAPSHOT_INTERVAL == 0:
            self._snapshot_policy()

        return {k: sum(v) / len(v) for k, v in stats.items()}

    def save(self, path: str) -> None:
        torch.save({"policy": self.policy.state_dict(), "optimizer": self.optimizer.state_dict(), "episodes": self.episodes}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.episodes = ckpt.get("episodes", 0)
