# Autonomous Multi-Agent Negotiation System for Dynamic Pricing

A hierarchical multi-agent negotiation system combining DQN-based strategy selection with Qwen-2.5 7B dialogue generation, deployed on Kubernetes with full CI/CD ~

Traditional pricing negotiation in e-commerce, supply chain, and algorithmic trading relies on rigid rule-based systems that cannot adapt to counterpart behavior in real time. This project addresses that gap by building a fully autonomous, hierarchical multi-agent negotiation system capable of conducting dynamic price negotiations end-to-end without human intervention.
The system decouples two distinct cognitive layers: a DQN-based strategy agent that selects high-level negotiation intents (anchor, concede, hold firm, explore, accept), and a fine-tuned Qwen-2.5 7B language model that translates those intents into contextually appropriate dialogue. This separation allows the system to learn negotiation strategy independently from language fluency, improving both sample efficiency and interpretability. Pydantic guardrails enforce hard bidding constraints in real time — preventing illegal offers, excessive concessions, and phase violations — while PPO self-play allows the dialogue agent to continuously improve by negotiating against past versions of itself across 5,000 simulated episodes.
The infrastructure is built for scale: Ray distributes simulation across Kubernetes workers on AWS/Azure, vLLM serves the language model at low latency, and Langfuse provides full trace-level observability per episode. A circuit breaker prevents cascading failures across tools. The result is a system that raised deal agreement rates from 42% to 61%, improved constraint satisfaction from 71% to 93%, and cut p95 response latency from 480ms to 148ms — all while maintaining stable performance under load through automated CI/CD regression gates.



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
