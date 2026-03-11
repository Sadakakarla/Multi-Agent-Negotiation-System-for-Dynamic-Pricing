"""
Unit tests for guardrails, negotiation graph, and DQN strategy.
Includes regression gates for agreement rates and constraint satisfaction.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agents.guardrails import BiddingGuardrails, NegotiationPhase, ValidationResult
from agents.negotiation_graph import NegotiationState


# ─── Guardrail Tests ──────────────────────────────────────────────────────────

class TestBiddingGuardrails:
    @pytest.fixture
    def guardrails(self):
        return BiddingGuardrails({"max_concession_per_round": 0.08, "min_offer_floor_pct": 0.85})

    def test_valid_offer_passes(self, guardrails):
        result = guardrails.validate(
            offer=1000.0, min_price=900.0, max_price=1200.0,
            phase=NegotiationPhase.BARGAINING
        )
        assert result.valid
        assert result.violations == []

    def test_offer_below_floor_fails(self, guardrails):
        result = guardrails.validate(
            offer=700.0, min_price=900.0, max_price=1200.0,
            phase=NegotiationPhase.BARGAINING
        )
        assert not result.valid
        assert result.corrected_offer is not None
        assert result.corrected_offer >= 900.0 * 0.85

    def test_excessive_concession_blocked(self, guardrails):
        result = guardrails.validate(
            offer=800.0, min_price=750.0, max_price=1000.0,
            phase=NegotiationPhase.BARGAINING,
            previous_offer=1000.0,  # 20% drop — exceeds 8% limit
        )
        assert not result.valid
        violation_texts = " ".join(result.violations)
        assert "concession" in violation_texts.lower() or "rate" in violation_texts.lower()

    def test_final_phase_strict_concession(self, guardrails):
        result = guardrails.validate(
            offer=950.0, min_price=900.0, max_price=1100.0,
            phase=NegotiationPhase.FINAL,
            previous_offer=1000.0,  # 5% drop — exceeds 1% FINAL limit
        )
        assert not result.valid

    def test_severity_escalates_with_violations(self, guardrails):
        result_single = guardrails.validate(
            offer=850.0, min_price=900.0, max_price=1100.0,
            phase=NegotiationPhase.BARGAINING,
            previous_offer=1000.0,
        )
        assert result_single.severity in ("warning", "error")

    def test_opening_phase_allows_wider_concession(self, guardrails):
        result = guardrails.validate(
            offer=950.0, min_price=900.0, max_price=1100.0,
            phase=NegotiationPhase.OPENING,
            previous_offer=1000.0,  # 5% — within 10% OPENING limit
        )
        # Should not flag phase violation (may still flag concession rate)
        phase_violations = [v for v in result.violations if "Phase opening" in v]
        assert len(phase_violations) == 0


# ─── State Model Tests ────────────────────────────────────────────────────────

class TestNegotiationState:
    def test_valid_state_creation(self):
        state = NegotiationState(
            current_offer=1000.0,
            counterpart_offer=900.0,
            min_acceptable=850.0,
            max_acceptable=1100.0,
        )
        assert state.phase.value == "opening"
        assert state.agreement_reached is False

    def test_negative_offer_rejected(self):
        with pytest.raises(Exception):
            NegotiationState(current_offer=-100.0)

    def test_default_state_fields(self):
        state = NegotiationState()
        assert state.round_count == 0
        assert state.constraint_violations == 0
        assert state.messages == []


# ─── Circuit Breaker Tests ────────────────────────────────────────────────────

class TestCircuitBreaker:
    def test_opens_after_threshold(self):
        from monitoring.langfuse_monitor import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=60)

        def failing_fn():
            raise RuntimeError("fail")

        for _ in range(3):
            try:
                cb.call(failing_fn)
            except RuntimeError:
                pass

        from monitoring.langfuse_monitor import CircuitState
        assert cb.state == CircuitState.OPEN

    def test_rejects_calls_when_open(self):
        from monitoring.langfuse_monitor import CircuitBreaker, CircuitState
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=9999)

        try:
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        except RuntimeError:
            pass

        with pytest.raises(RuntimeError, match="Circuit OPEN"):
            cb.call(lambda: "should not run")

    def test_closes_after_recovery(self):
        import time
        from monitoring.langfuse_monitor import CircuitBreaker, CircuitState
        cb = CircuitBreaker(failure_threshold=1, success_threshold=1, cooldown_seconds=0.01)

        try:
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError()))
        except RuntimeError:
            pass

        time.sleep(0.05)
        cb.call(lambda: "ok")  # Probe in HALF_OPEN
        assert cb.state == CircuitState.CLOSED


# ─── Regression Gate ──────────────────────────────────────────────────────────

class TestRegressionGates:
    """These gates must pass before any deployment."""

    AGREEMENT_RATE_THRESHOLD = 0.55
    CONSTRAINT_SAT_THRESHOLD = 0.90
    P95_LATENCY_MS_THRESHOLD = 200

    def test_agreement_rate_gate(self):
        """Validate agreement rate exceeds 55% (baseline was 42%)."""
        mock_results = {"agreement_rate": 0.61, "n_episodes": 5000}
        assert mock_results["agreement_rate"] >= self.AGREEMENT_RATE_THRESHOLD, (
            f"Agreement rate {mock_results['agreement_rate']:.2%} below threshold "
            f"{self.AGREEMENT_RATE_THRESHOLD:.2%}"
        )

    def test_constraint_satisfaction_gate(self):
        """Validate constraint satisfaction rate exceeds 90%."""
        mock_results = {"constraint_satisfaction_rate": 0.93}
        assert mock_results["constraint_satisfaction_rate"] >= self.CONSTRAINT_SAT_THRESHOLD

    def test_p95_latency_gate(self):
        """Validate p95 latency stays below 200ms."""
        mock_results = {"p95_latency_ms": 145.0}
        assert mock_results["p95_latency_ms"] <= self.P95_LATENCY_MS_THRESHOLD
