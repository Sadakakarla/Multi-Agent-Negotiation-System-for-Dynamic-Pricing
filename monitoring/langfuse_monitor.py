"""
Langfuse monitoring for negotiation traces, latency, and agreement metrics.
Includes circuit breaker to protect downstream services.
"""
from __future__ import annotations

import asyncio
import functools
import time
from enum import Enum
from typing import Any, Callable

from langfuse import Langfuse
from pydantic import BaseModel


# ─── Circuit Breaker ──────────────────────────────────────────────────────────

class CircuitState(str, Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing — reject calls
    HALF_OPEN = "half_open"  # Probing recovery


class CircuitBreaker:
    """
    Tool-correctness circuit breaker.
    Opens after threshold failures; auto-resets after cooldown.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        cooldown_seconds: float = 60.0,
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.cooldown = cooldown_seconds
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.successes = 0
        self.opened_at: float = 0.0

    def call(self, fn: Callable, *args, **kwargs) -> Any:
        if self.state == CircuitState.OPEN:
            if time.monotonic() - self.opened_at >= self.cooldown:
                self.state = CircuitState.HALF_OPEN
            else:
                raise RuntimeError("Circuit OPEN — downstream tool unavailable")

        try:
            result = fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure()
            raise

    async def async_call(self, fn: Callable, *args, **kwargs) -> Any:
        if self.state == CircuitState.OPEN:
            if time.monotonic() - self.opened_at >= self.cooldown:
                self.state = CircuitState.HALF_OPEN
            else:
                raise RuntimeError("Circuit OPEN — downstream tool unavailable")
        try:
            result = await fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        if self.state == CircuitState.HALF_OPEN:
            self.successes += 1
            if self.successes >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failures = 0
                self.successes = 0

    def _on_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.opened_at = time.monotonic()

    @property
    def is_available(self) -> bool:
        return self.state != CircuitState.OPEN


# ─── Monitoring ───────────────────────────────────────────────────────────────

class NegotiationMonitor:
    """Wraps Langfuse for trace-level observability of negotiation episodes."""

    def __init__(self, cfg: dict):
        self.langfuse = Langfuse(
            public_key=cfg["langfuse_public_key"],
            secret_key=cfg["langfuse_secret_key"],
            host=cfg.get("langfuse_host", "https://cloud.langfuse.com"),
        )
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=cfg.get("cb_failure_threshold", 5),
            cooldown_seconds=cfg.get("cb_cooldown_seconds", 60),
        )

    def trace_episode(self, episode_id: int, scenario: dict) -> Any:
        return self.langfuse.trace(
            name=f"negotiation-episode-{episode_id}",
            metadata={
                "episode_id": episode_id,
                "seller_ask": scenario.get("seller_ask"),
                "buyer_bid": scenario.get("buyer_bid"),
            },
            tags=["negotiation", "pricing"],
        )

    def log_intent(self, trace: Any, intent: str, obs: dict, round_num: int) -> None:
        trace.span(
            name="dqn-intent-selection",
            input=obs,
            output={"intent": intent},
            metadata={"round": round_num},
        )

    def log_utterance(self, trace: Any, utterance: str, offer: float, latency_ms: float) -> None:
        trace.span(
            name="dialogue-generation",
            output={"utterance": utterance, "offer": offer},
            metadata={"latency_ms": latency_ms},
        )

    def log_episode_result(self, trace: Any, result: dict) -> None:
        trace.update(
            output=result,
            metadata={
                "agreement_reached": result.get("agreement_reached"),
                "final_offer": result.get("final_offer"),
                "rounds": result.get("rounds"),
                "constraint_violations": result.get("constraint_violations"),
            },
        )
        self.langfuse.score(
            trace_id=trace.id,
            name="agreement_rate",
            value=1.0 if result.get("agreement_reached") else 0.0,
        )
        self.langfuse.score(
            trace_id=trace.id,
            name="constraint_satisfaction",
            value=max(0, 1 - result.get("constraint_violations", 0) / 10),
        )

    def flush(self) -> None:
        self.langfuse.flush()
