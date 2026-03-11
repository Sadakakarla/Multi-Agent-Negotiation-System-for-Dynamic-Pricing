"""
Hierarchical Negotiation Policy via LangGraph.
Decouples strategy (DQN intent selection) from dialogue (Qwen-2.5 7B SFT+PPO).
"""
from __future__ import annotations

import asyncio
from enum import Enum
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator

from agents.dqn_strategy import DQNStrategyAgent
from agents.dialogue_agent import DialogueAgent
from agents.guardrails import BiddingGuardrails


# ─── State ────────────────────────────────────────────────────────────────────

class NegotiationPhase(str, Enum):
    OPENING = "opening"
    BARGAINING = "bargaining"
    CLOSING = "closing"
    FINAL = "final"


class NegotiationState(BaseModel):
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    phase: NegotiationPhase = NegotiationPhase.OPENING
    current_offer: float = 0.0
    counterpart_offer: float = 0.0
    min_acceptable: float = 0.0
    max_acceptable: float = 0.0
    intent: str = ""
    episode: int = 0
    agreement_reached: bool = False
    constraint_violations: int = 0
    round_count: int = 0

    @field_validator("current_offer")
    @classmethod
    def offer_must_be_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Offer must be non-negative")
        return v


# ─── Nodes ────────────────────────────────────────────────────────────────────

class NegotiationGraph:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.strategy_agent = DQNStrategyAgent(cfg["dqn"])
        self.dialogue_agent = DialogueAgent(cfg["dialogue"])
        self.guardrails = BiddingGuardrails(cfg["guardrails"])
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        builder = StateGraph(NegotiationState)

        builder.add_node("select_intent", self._select_intent)
        builder.add_node("generate_utterance", self._generate_utterance)
        builder.add_node("apply_guardrails", self._apply_guardrails)
        builder.add_node("update_offer", self._update_offer)
        builder.add_node("check_terminal", self._check_terminal)

        builder.add_edge(START, "select_intent")
        builder.add_edge("select_intent", "generate_utterance")
        builder.add_edge("generate_utterance", "apply_guardrails")
        builder.add_edge("apply_guardrails", "update_offer")
        builder.add_edge("update_offer", "check_terminal")
        builder.add_conditional_edges(
            "check_terminal",
            self._route_terminal,
            {"continue": "select_intent", "end": END},
        )
        return builder.compile()

    # ── Node implementations ─────────────────────────────────────────────────

    async def _select_intent(self, state: NegotiationState) -> dict:
        """DQN selects high-level negotiation intent."""
        obs = self._build_obs(state)
        intent = await self.strategy_agent.select_intent(obs)
        return {"intent": intent}

    async def _generate_utterance(self, state: NegotiationState) -> dict:
        """Qwen-2.5 7B generates dialogue conditioned on intent."""
        utterance = await self.dialogue_agent.generate(
            history=state.messages,
            intent=state.intent,
            current_offer=state.current_offer,
            phase=state.phase,
        )
        return {"messages": [AIMessage(content=utterance)]}

    async def _apply_guardrails(self, state: NegotiationState) -> dict:
        """Pydantic guardrails enforce real-time bidding rules."""
        result = self.guardrails.validate(
            offer=state.current_offer,
            min_price=state.min_acceptable,
            max_price=state.max_acceptable,
            phase=state.phase,
        )
        violations = state.constraint_violations + (0 if result.valid else 1)
        return {"constraint_violations": violations}

    async def _update_offer(self, state: NegotiationState) -> dict:
        """Extract numeric offer from last AI message and update phase."""
        last_msg = state.messages[-1].content if state.messages else ""
        new_offer = self._extract_offer(last_msg, state.current_offer)
        phase = self._advance_phase(state)
        return {"current_offer": new_offer, "phase": phase, "round_count": state.round_count + 1}

    async def _check_terminal(self, state: NegotiationState) -> dict:
        """Determine if negotiation has concluded."""
        zopa = abs(state.current_offer - state.counterpart_offer)
        agreed = zopa < self.cfg.get("agreement_threshold", 0.05) * state.counterpart_offer
        return {"agreement_reached": agreed}

    # ── Routing ──────────────────────────────────────────────────────────────

    def _route_terminal(self, state: NegotiationState) -> Literal["continue", "end"]:
        if state.agreement_reached:
            return "end"
        if state.round_count >= self.cfg.get("max_rounds", 20):
            return "end"
        return "continue"

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _build_obs(self, state: NegotiationState) -> dict:
        return {
            "phase": state.phase.value,
            "current_offer": state.current_offer,
            "counterpart_offer": state.counterpart_offer,
            "round": state.round_count,
            "zopa": abs(state.current_offer - state.counterpart_offer),
        }

    def _extract_offer(self, text: str, fallback: float) -> float:
        import re
        match = re.search(r"\$?([\d,]+(?:\.\d{1,2})?)", text.replace(",", ""))
        return float(match.group(1)) if match else fallback

    def _advance_phase(self, state: NegotiationState) -> NegotiationPhase:
        thresholds = {10: NegotiationPhase.BARGAINING, 16: NegotiationPhase.CLOSING}
        for round_threshold, phase in sorted(thresholds.items()):
            if state.round_count < round_threshold:
                return phase
        return NegotiationPhase.FINAL

    async def run_episode(self, initial_state: NegotiationState) -> NegotiationState:
        result = await self.graph.ainvoke(initial_state)
        return result
