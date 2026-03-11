"""
Pydantic-based guardrails for real-time bidding rule enforcement.
Validates offers, enforces ZOPA constraints, and prevents rule violations.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class NegotiationPhase(str, Enum):
    OPENING = "opening"
    BARGAINING = "bargaining"
    CLOSING = "closing"
    FINAL = "final"


# ─── Validation Models ────────────────────────────────────────────────────────

class BidRequest(BaseModel):
    offer: float = Field(..., gt=0, description="Proposed offer amount")
    min_price: float = Field(..., gt=0)
    max_price: float = Field(..., gt=0)
    phase: NegotiationPhase
    previous_offer: float = Field(default=0.0, ge=0)
    round_number: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def max_must_exceed_min(self) -> "BidRequest":
        if self.max_price <= self.min_price:
            raise ValueError("max_price must exceed min_price")
        return self

    @model_validator(mode="after")
    def offer_within_bounds(self) -> "BidRequest":
        margin = 0.15  # 15% tolerance outside ZOPA
        lower = self.min_price * (1 - margin)
        upper = self.max_price * (1 + margin)
        if not (lower <= self.offer <= upper):
            raise ValueError(
                f"Offer {self.offer:.2f} is outside acceptable range "
                f"[{lower:.2f}, {upper:.2f}]"
            )
        return self


class ValidationResult(BaseModel):
    valid: bool
    violations: list[str] = Field(default_factory=list)
    corrected_offer: float | None = None
    severity: str = "none"  # none | warning | error


class PhaseTransitionRule(BaseModel):
    from_phase: NegotiationPhase
    to_phase: NegotiationPhase
    min_rounds: int
    max_concession_pct: float


# ─── Guardrails Engine ────────────────────────────────────────────────────────

class BiddingGuardrails:
    """
    Enforces bidding rules at inference time.
    All rules are declarative Pydantic models — no ad hoc conditionals.
    """

    PHASE_RULES: list[PhaseTransitionRule] = [
        PhaseTransitionRule(from_phase=NegotiationPhase.OPENING,    to_phase=NegotiationPhase.BARGAINING, min_rounds=2,  max_concession_pct=0.10),
        PhaseTransitionRule(from_phase=NegotiationPhase.BARGAINING, to_phase=NegotiationPhase.CLOSING,    min_rounds=5,  max_concession_pct=0.05),
        PhaseTransitionRule(from_phase=NegotiationPhase.CLOSING,    to_phase=NegotiationPhase.FINAL,      min_rounds=10, max_concession_pct=0.02),
    ]

    MAX_CONCESSION_PER_ROUND = 0.08  # 8% max drop per round
    MIN_OFFER_FLOOR_PCT = 0.85       # Never go below 85% of min_price

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.max_concession = cfg.get("max_concession_per_round", self.MAX_CONCESSION_PER_ROUND)
        self.floor_pct = cfg.get("min_offer_floor_pct", self.MIN_OFFER_FLOOR_PCT)

    def validate(
        self,
        offer: float,
        min_price: float,
        max_price: float,
        phase: NegotiationPhase,
        previous_offer: float = 0.0,
        round_number: int = 0,
    ) -> ValidationResult:
        violations: list[str] = []
        corrected = offer

        # 1. Structural validation via Pydantic
        try:
            BidRequest(
                offer=offer,
                min_price=min_price,
                max_price=max_price,
                phase=phase,
                previous_offer=previous_offer,
                round_number=round_number,
            )
        except Exception as exc:
            violations.append(f"Structural: {exc}")
            corrected = self._clamp_to_bounds(offer, min_price, max_price)

        # 2. Concession rate check
        if previous_offer > 0 and offer < previous_offer:
            concession_pct = (previous_offer - offer) / previous_offer
            if concession_pct > self.max_concession:
                violations.append(
                    f"Concession rate {concession_pct:.1%} exceeds limit {self.max_concession:.1%}"
                )
                corrected = previous_offer * (1 - self.max_concession)

        # 3. Absolute floor enforcement
        floor = min_price * self.floor_pct
        if offer < floor:
            violations.append(f"Offer {offer:.2f} below absolute floor {floor:.2f}")
            corrected = floor

        # 4. Phase-appropriate concession bounds
        phase_violation = self._check_phase_bounds(offer, previous_offer, phase)
        if phase_violation:
            violations.append(phase_violation)

        severity = "none" if not violations else ("warning" if len(violations) == 1 else "error")

        return ValidationResult(
            valid=len(violations) == 0,
            violations=violations,
            corrected_offer=corrected if violations else None,
            severity=severity,
        )

    def _clamp_to_bounds(self, offer: float, min_price: float, max_price: float) -> float:
        return max(min_price * self.floor_pct, min(offer, max_price * 1.15))

    def _check_phase_bounds(
        self, offer: float, previous_offer: float, phase: NegotiationPhase
    ) -> str | None:
        phase_limits = {
            NegotiationPhase.OPENING: 0.10,
            NegotiationPhase.BARGAINING: 0.06,
            NegotiationPhase.CLOSING: 0.03,
            NegotiationPhase.FINAL: 0.01,
        }
        limit = phase_limits.get(phase, 0.10)
        if previous_offer > 0 and offer < previous_offer:
            concession = (previous_offer - offer) / previous_offer
            if concession > limit:
                return f"Phase {phase.value} allows max {limit:.1%} concession; got {concession:.1%}"
        return None
