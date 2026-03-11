"""
Distributed negotiation simulation using Ray.
Runs parallel episodes across workers for fast RL training.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import ray

from agents.negotiation_graph import NegotiationGraph, NegotiationState, NegotiationPhase


@dataclass
class EpisodeResult:
    episode_id: int
    agreement_reached: bool
    final_offer: float
    rounds: int
    constraint_violations: int
    duration_ms: float
    deal_value: float = 0.0


@ray.remote
class NegotiationWorker:
    """Ray remote actor — one per CPU core on the cluster."""

    def __init__(self, cfg: dict, worker_id: int):
        self.worker_id = worker_id
        self.graph = NegotiationGraph(cfg)
        self.episodes_run = 0

    async def run_episode(self, episode_id: int, scenario: dict) -> EpisodeResult:
        start = time.monotonic()
        state = NegotiationState(
            current_offer=scenario["seller_ask"],
            counterpart_offer=scenario["buyer_bid"],
            min_acceptable=scenario["reserve_price"],
            max_acceptable=scenario["target_price"],
            episode=episode_id,
        )
        result_state = await self.graph.run_episode(state)
        duration_ms = (time.monotonic() - start) * 1000
        self.episodes_run += 1

        deal_value = 0.0
        if result_state["agreement_reached"]:
            deal_value = result_state.get("current_offer", 0.0)

        return EpisodeResult(
            episode_id=episode_id,
            agreement_reached=result_state.get("agreement_reached", False),
            final_offer=result_state.get("current_offer", 0.0),
            rounds=result_state.get("round_count", 0),
            constraint_violations=result_state.get("constraint_violations", 0),
            duration_ms=duration_ms,
            deal_value=deal_value,
        )

    def get_stats(self) -> dict:
        return {"worker_id": self.worker_id, "episodes_run": self.episodes_run}


class DistributedSimulator:
    """
    Orchestrates distributed negotiation simulations across Kubernetes pods.
    Uses Ray for actor management and TTL caching for scenario reuse.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.n_workers = cfg.get("n_workers", 8)
        self._workers: list[Any] = []
        self._scenario_cache: dict[str, dict] = {}  # TTL cache
        self._cache_ttl = cfg.get("cache_ttl_seconds", 300)
        self._cache_timestamps: dict[str, float] = {}

    def init_workers(self) -> None:
        if not ray.is_initialized():
            ray.init(address=self.cfg.get("ray_address", "auto"), ignore_reinit_error=True)
        self._workers = [
            NegotiationWorker.remote(self.cfg["negotiation"], worker_id=i)
            for i in range(self.n_workers)
        ]

    async def run_parallel_episodes(
        self, n_episodes: int, scenario_fn: Any
    ) -> list[EpisodeResult]:
        """Run n_episodes in parallel, distributing across workers."""
        tasks = []
        for ep_id in range(n_episodes):
            worker = self._workers[ep_id % self.n_workers]
            scenario = self._get_or_fetch_scenario(scenario_fn, ep_id)
            tasks.append(worker.run_episode.remote(ep_id, scenario))

        results = await asyncio.gather(*[asyncio.wrap_future(t.future()) for t in tasks], return_exceptions=True)
        return [r for r in results if isinstance(r, EpisodeResult)]

    def run_episodes_sync(self, n_episodes: int, scenario_fn: Any) -> list[EpisodeResult]:
        """Synchronous wrapper for non-async contexts."""
        tasks = []
        for ep_id in range(n_episodes):
            worker = self._workers[ep_id % self.n_workers]
            scenario = self._get_or_fetch_scenario(scenario_fn, ep_id)
            tasks.append(worker.run_episode.remote(ep_id, scenario))
        return ray.get(tasks)

    def _get_or_fetch_scenario(self, scenario_fn: Any, ep_id: int) -> dict:
        key = f"scenario_{ep_id % 100}"  # Reuse 100 base scenarios
        now = time.monotonic()

        if key in self._scenario_cache:
            if now - self._cache_timestamps.get(key, 0) < self._cache_ttl:
                return self._scenario_cache[key]

        scenario = scenario_fn(ep_id)
        self._scenario_cache[key] = scenario
        self._cache_timestamps[key] = now
        return scenario

    def aggregate_metrics(self, results: list[EpisodeResult]) -> dict:
        if not results:
            return {}
        n = len(results)
        agreements = sum(r.agreement_reached for r in results)
        violations = sum(r.constraint_violations for r in results)

        durations = sorted(r.duration_ms for r in results)
        p95_idx = int(0.95 * n)

        return {
            "n_episodes": n,
            "agreement_rate": agreements / n,
            "avg_rounds": sum(r.rounds for r in results) / n,
            "total_constraint_violations": violations,
            "constraint_satisfaction_rate": 1 - (violations / (n * 20)),
            "p95_latency_ms": durations[p95_idx] if durations else 0,
            "avg_deal_value": sum(r.deal_value for r in results if r.agreement_reached) / max(agreements, 1),
        }

    def shutdown(self) -> None:
        if ray.is_initialized():
            ray.shutdown()
