# Architecture

Automation uses an agent orchestration core (`AgentSystem`) plus DQN training services. Tasks are persisted via the configured task store and executed through role-based agents.

## Data Flow
1. Task creation (`create_task`)
2. Assignment via orchestrator/explicit target
3. Execution (`execute_task`)
4. Metrics/tracing/logging emission
5. Persistence and observability snapshots

## Key Components
- `src/agents/super_agentic_agents.py`
- `src/models/neural_network.py`
- `src/training/dqn_service.py`
- `src/observability/*`
- `src/monitoring/*`
- `src/resilience/*`
