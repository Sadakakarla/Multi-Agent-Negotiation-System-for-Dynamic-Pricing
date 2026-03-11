"""
Dialogue agent using Qwen-2.5 7B via vLLM inference.
Supports SFT fine-tuning and PPO self-play alignment.
"""
from __future__ import annotations

import asyncio
from typing import Any

import httpx
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from agents.negotiation_graph import NegotiationPhase


SYSTEM_PROMPT = """You are a skilled negotiation agent optimizing for deal closure.
Your goal is to reach an agreement within acceptable price bounds while maintaining rapport.

Guidelines:
- Be concise and professional
- When making offers, state the price explicitly (e.g., "$1,250")
- Match the intent provided: {intent}
- Current phase: {phase}
- Your current offer: ${current_offer:.2f}

Respond with a single negotiation utterance only."""


class DialogueConfig(BaseModel):
    vllm_base_url: str = "http://localhost:8000"
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    temperature: float = 0.7
    max_tokens: int = 256
    top_p: float = 0.9
    timeout: float = 10.0


class PPORewardSignal(BaseModel):
    agreement_bonus: float = 10.0
    constraint_penalty: float = -5.0
    concession_rate_weight: float = 0.3
    rapport_score_weight: float = 0.2


class DialogueAgent:
    def __init__(self, cfg: dict):
        self.cfg = DialogueConfig(**cfg)
        self.reward_cfg = PPORewardSignal(**cfg.get("ppo_reward", {}))
        self.client = httpx.AsyncClient(base_url=self.cfg.vllm_base_url, timeout=self.cfg.timeout)

    async def generate(
        self,
        history: list[BaseMessage],
        intent: str,
        current_offer: float,
        phase: NegotiationPhase,
    ) -> str:
        system = SYSTEM_PROMPT.format(
            intent=intent,
            phase=phase.value,
            current_offer=current_offer,
        )
        messages = self._format_messages(history, system)

        payload = {
            "model": self.cfg.model_name,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "top_p": self.cfg.top_p,
        }
        try:
            response = await self.client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except httpx.HTTPError as e:
            return f"I'd like to discuss the offer of ${current_offer:.2f} further."

    def compute_ppo_reward(
        self,
        agreement_reached: bool,
        constraint_violations: int,
        concession_delta: float,
        rapport_score: float,
    ) -> float:
        """Compute shaped reward signal for PPO self-play training."""
        reward = 0.0
        if agreement_reached:
            reward += self.reward_cfg.agreement_bonus
        reward += constraint_violations * self.reward_cfg.constraint_penalty
        reward += concession_delta * self.reward_cfg.concession_rate_weight
        reward += rapport_score * self.reward_cfg.rapport_score_weight
        return reward

    def _format_messages(self, history: list[BaseMessage], system: str) -> list[dict]:
        messages = [{"role": "system", "content": system}]
        for msg in history[-10:]:  # Last 10 turns for context window
            role = "assistant" if msg.__class__.__name__ == "AIMessage" else "user"
            messages.append({"role": role, "content": msg.content})
        return messages

    async def close(self) -> None:
        await self.client.aclose()
