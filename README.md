# Autonomous Multi-Agent Negotiation System for Dynamic Pricing

A hierarchical negotiation system that combines RL-based strategy selection with LLM-driven (Qwen-2.5 7B) dialogue generation for dynamic pricing at production scale.

Traditional pricing negotiation in e-commerce, supply chain, and algorithmic trading often relies on rigid rule-based systems that cannot adapt to counterpart behavior in real time. This project builds a fully autonomous hierarchical multi-agent negotiation system that can conduct dynamic price negotiations end to end without human intervention.

The architecture separates strategy from dialogue into two layers:
A Double DQN agent selects high-level negotiation intents such as anchor, concede, hold firm, explore, and accept, while a fine-tuned Qwen-2.5 7B model converts those intents into context-aware responses. This design improves sample efficiency, interpretability, and control over negotiation behavior. Pydantic guardrails enforce hard bidding constraints in real time by blocking invalid offers, excessive concessions, and phase violations. PPO self-play further improves performance by having the agent negotiate against past versions of itself across 5,000 simulated episodes.

The system is built for production-scale deployment. Ray distributes simulation workloads across Kubernetes workers on AWS and Azure, vLLM serves the language model with low-latency inference, and Langfuse provides trace-level observability for every episode. Circuit breakers prevent cascading tool failures, while CI/CD regression gates enforce performance thresholds before deployment. 

Results: In evaluation, the system increased deal agreement rates from 42% to 61%, improved constraint satisfaction from 71% to 93%, and reduced p95 latency from 480 ms to 148 ms.


## Results
| Metric | Baseline | This System |
|---|---|---|
| Deal Agreement Rate | 42% | **61%** |
| Constraint Satisfaction | 71% | **93%** |
| Episodes Evaluated | 1,000 | **5,000** |
| p95 Latency | 480ms | **148ms** |

## Architecture

```
LangGraph Orchestration
├── DQN Strategy Agent      ← High-level intent selection (Double DQN + PER)
├── Qwen-2.5 7B Dialogue    ← Utterance generation (SFT + PPO self-play)
├── Pydantic Guardrails     ← Real-time bidding rule enforcement
└── Langfuse Monitor        ← Trace-level observability + circuit breaker
```

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Train (local)
python scripts/train.py --config configs/default.yaml

# Simulate 500 episodes
python scripts/run_simulation.py --episodes 500
```

## Kubernetes Deployment

```bash
# Apply manifests
kubectl apply -f infrastructure/k8s-deployment.yaml -n negotiation-system

# Watch rollout
kubectl rollout status deployment/negotiation-worker -n negotiation-system
```

## Project Structure

```
├── agents/
│   ├── negotiation_graph.py    # LangGraph orchestration
│   ├── dqn_strategy.py         # Double DQN + prioritized replay
│   ├── dialogue_agent.py       # Qwen-2.5 7B via vLLM
│   └── guardrails.py           # Pydantic bidding rule enforcement
├── environment/
│   └── distributed_sim.py      # Ray-based parallel simulation
├── training/
│   └── ppo_selfplay.py         # PPO self-play trainer
├── monitoring/
│   └── langfuse_monitor.py     # Langfuse + circuit breaker
├── infrastructure/
│   └── k8s-deployment.yaml     # K8s + HPA manifests
├── tests/
│   └── test_negotiation.py     # Unit + regression gate tests
├── configs/
│   └── default.yaml
├── .github/workflows/
│   └── ci-cd.yml               # CI/CD with regression gates
└── Dockerfile
```

## CI/CD Gates

The pipeline enforces these gates before any deployment:
- **Agreement rate ≥ 55%** (production: 61%)
- **Constraint satisfaction ≥ 90%**
- **p95 latency ≤ 200ms**
- All unit + guardrail tests passing
