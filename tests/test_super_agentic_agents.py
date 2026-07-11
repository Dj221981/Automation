"""
Tests for the production-ready super_agentic_agents framework.

Covers:
- Agent creation and capability registration
- Task creation, assignment, and execution
- Retry logic (mock a failing act() that succeeds on 3rd attempt)
- AgentConfig and TaskConfig validation
- AgentSystem.get_health() output structure
"""

import asyncio
import uuid
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from src.agents.super_agentic_agents import (
    AgentCapability,
    AgentConfig,
    AgentRole,
    AgentStatus,
    AgentSystem,
    AnalyzerAgent,
    BaseAgent,
    ExecutorAgent,
    LearnerAgent,
    OrchestratorAgent,
    Task,
    TaskConfig,
    TaskPriority,
    AgentFactory,
    AgentMemory,
    StructuredLogger,
)


# ============================================================================
# Helpers
# ============================================================================

def make_task(description: str = "Test task", priority: TaskPriority = TaskPriority.NORMAL) -> Task:
    return Task(description=description, priority=priority, parameters={"key": "value"})


# ============================================================================
# AgentConfig validation
# ============================================================================

class TestAgentConfig:
    def test_valid_config(self):
        cfg = AgentConfig(name="MyAgent", max_capabilities=10, max_retries=3)
        assert cfg.name == "MyAgent"
        assert cfg.max_capabilities == 10
        assert cfg.max_retries == 3

    def test_name_too_long(self):
        with pytest.raises((ValueError, Exception)):
            AgentConfig(name="x" * 101)

    def test_empty_name(self):
        with pytest.raises((ValueError, Exception)):
            AgentConfig(name="")

    def test_max_capabilities_out_of_range(self):
        with pytest.raises((ValueError, Exception)):
            AgentConfig(name="A", max_capabilities=0)
        with pytest.raises((ValueError, Exception)):
            AgentConfig(name="A", max_capabilities=201)

    def test_max_retries_out_of_range(self):
        with pytest.raises((ValueError, Exception)):
            AgentConfig(name="A", max_retries=-1)
        with pytest.raises((ValueError, Exception)):
            AgentConfig(name="A", max_retries=11)


# ============================================================================
# TaskConfig validation
# ============================================================================

class TestTaskConfig:
    def test_valid_task_config(self):
        cfg = TaskConfig(
            description="Do something",
            priority=TaskPriority.HIGH,
            parameters={"a": 1},
            dependencies=[],
        )
        assert cfg.description == "Do something"

    def test_empty_description(self):
        with pytest.raises((ValueError, Exception)):
            TaskConfig(description="")

    def test_invalid_dependency_uuid(self):
        with pytest.raises((ValueError, Exception)):
            TaskConfig(description="Task", dependencies=["not-a-uuid"])

    def test_valid_uuid_dependency(self):
        valid_uuid = str(uuid.uuid4())
        cfg = TaskConfig(description="Task", dependencies=[valid_uuid])
        assert cfg.dependencies == [valid_uuid]


# ============================================================================
# Agent creation and capability registration
# ============================================================================

class TestAgentCreation:
    def test_executor_agent_defaults(self):
        agent = ExecutorAgent("Exec-1")
        assert agent.name == "Exec-1"
        assert agent.role == AgentRole.EXECUTOR
        assert agent.status == AgentStatus.IDLE

    def test_analyzer_agent_defaults(self):
        agent = AnalyzerAgent("Analyzer-1")
        assert agent.role == AgentRole.ANALYZER

    def test_learner_agent_defaults(self):
        agent = LearnerAgent("Learner-1")
        assert agent.role == AgentRole.LEARNER

    def test_orchestrator_agent_defaults(self):
        agent = OrchestratorAgent("Orch-1")
        assert agent.role == AgentRole.ORCHESTRATOR

    def test_register_capability(self):
        agent = ExecutorAgent("Exec-2")
        cap = AgentCapability(name="my_cap", description="test cap", confidence_score=0.9)
        result = agent.register_capability(cap)
        assert result is True
        assert "my_cap" in agent.list_capabilities()

    def test_max_capabilities_limit(self):
        agent = ExecutorAgent("Exec-3", max_capabilities=2)
        for i in range(2):
            agent.register_capability(AgentCapability(name=f"cap_{i}", description="cap"))
        # Third registration should fail
        result = agent.register_capability(AgentCapability(name="cap_3", description="cap"))
        assert result is False

    def test_agent_repr(self):
        agent = ExecutorAgent("Exec-repr")
        assert "ExecutorAgent" in repr(agent)
        assert "Exec-repr" in repr(agent)

    def test_get_status_keys(self):
        agent = ExecutorAgent("Exec-status")
        status = agent.get_status()
        for key in ("id", "name", "role", "status", "capabilities", "performance"):
            assert key in status


# ============================================================================
# Task creation and assignment
# ============================================================================

class TestTaskManagement:
    def test_create_task(self):
        system = AgentSystem("TestSys")
        task = system.create_task("Test", parameters={"x": 1})
        assert task.description == "Test"
        assert task.id  # has a UUID

    def test_task_assignment(self):
        agent = ExecutorAgent("Exec-assign")
        task = make_task()
        result = agent.assign_task(task)
        assert result is True
        assert task.id in agent.active_tasks
        assert task.assigned_to == agent.id

    def test_task_to_dict(self):
        task = make_task()
        d = task.to_dict()
        assert d["status"] == "pending"
        assert "id" in d


# ============================================================================
# Async task execution
# ============================================================================

@pytest.mark.asyncio
class TestAsyncExecution:
    async def test_execute_task_success(self):
        agent = ExecutorAgent("Exec-async")
        task = make_task("Async task")
        agent.assign_task(task)
        result = await agent.execute_task(task)
        assert result["execution"] == "successful"
        assert task.status == "completed"
        assert agent.status == AgentStatus.IDLE

    async def test_execute_task_sets_duration(self):
        agent = ExecutorAgent("Exec-duration")
        task = make_task()
        agent.assign_task(task)
        await agent.execute_task(task)
        assert "duration_ms" in task.metadata
        assert task.metadata["duration_ms"] >= 0

    async def test_run_tasks_concurrent(self):
        system = AgentSystem("ConcurrentSys")
        agents = [ExecutorAgent(f"Exec-{i}") for i in range(3)]
        for a in agents:
            system.add_agent(a)

        pairs = []
        for agent in agents:
            task = make_task(f"Task for {agent.name}")
            agent.assign_task(task)
            pairs.append((agent, task))

        results = await system.run_tasks(pairs)
        assert len(results) == 3
        for r in results:
            assert not isinstance(r, Exception)

    async def test_analyzer_agent_execution(self):
        agent = AnalyzerAgent("Analyze-async")
        task = make_task("Analyze data")
        agent.assign_task(task)
        result = await agent.execute_task(task)
        assert result["analysis_complete"] is True

    async def test_learner_agent_execution(self):
        agent = LearnerAgent("Learn-async")
        task = make_task("Learn something")
        agent.assign_task(task)
        result = await agent.execute_task(task)
        assert "learning" in result

    async def test_orchestrator_agent_execution(self):
        agent = OrchestratorAgent("Orch-async")
        task = make_task("Orchestrate")
        agent.assign_task(task)
        result = await agent.execute_task(task)
        assert result["status"] == "orchestration_complete"


# ============================================================================
# Retry logic
# ============================================================================

class _FlakyExecutorAgent(ExecutorAgent):
    """An executor whose act() fails the first two times, then succeeds."""
    def __init__(self, name: str = "Flaky"):
        super().__init__(name, max_retries=3, retry_delay=0.01)
        self._call_count = 0

    async def act(self, decision: Dict[str, Any]) -> Any:
        self._call_count += 1
        if self._call_count < 3:
            raise RuntimeError(f"Simulated failure (attempt {self._call_count})")
        return {"execution": "successful", "parameters_processed": {}}


@pytest.mark.asyncio
class TestRetryLogic:
    async def test_retries_then_succeeds(self):
        agent = _FlakyExecutorAgent()
        task = make_task("Retry task")
        agent.assign_task(task)
        result = await agent.execute_task(task)
        assert result["execution"] == "successful"
        assert agent._call_count == 3
        assert task.status == "completed"
        assert agent.status == AgentStatus.IDLE

    async def test_exhausted_retries_raises(self):
        class AlwaysFailAgent(ExecutorAgent):
            def __init__(self):
                super().__init__("AlwaysFail", max_retries=2, retry_delay=0.01)

            async def act(self, decision: Dict[str, Any]) -> Any:
                raise ValueError("Permanent failure")

        agent = AlwaysFailAgent()
        task = make_task("Always fail")
        agent.assign_task(task)

        with pytest.raises(ValueError, match="Permanent failure"):
            await agent.execute_task(task)

        assert agent.status == AgentStatus.ERROR
        assert task.status == "failed"
        assert "Permanent failure" in (task.error or "")


# ============================================================================
# LLM client integration (mocked)
# ============================================================================

@pytest.mark.asyncio
class TestLLMIntegration:
    async def test_orchestrator_uses_llm_when_provided(self):
        """When llm_client is set, OrchestratorAgent.think() calls the API."""
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = (
            '{"analysis": "LLM plan", "priority": "high", "execution_strategy": "sequential"}'
        )
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        agent = OrchestratorAgent("LLM-Orch", llm_client=mock_client)
        result = await agent.think({"task": "do something"})
        assert result["analysis"] == "LLM plan"
        mock_client.chat.completions.create.assert_called_once()

    async def test_orchestrator_fallback_without_llm(self):
        agent = OrchestratorAgent("NoLLM-Orch")
        result = await agent.think({})
        assert "analysis" in result
        assert "execution_strategy" in result

    async def test_analyzer_uses_llm_when_provided(self):
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = (
            '{"data_received": true, "analysis_type": "LLM", '
            '"insights_generated": true, "summary": "LLM analysis"}'
        )
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        agent = AnalyzerAgent("LLM-Analyzer", llm_client=mock_client)
        result = await agent.think({"data": [1, 2, 3]})
        assert result["analysis_type"] == "LLM"

    async def test_analyzer_fallback_without_llm(self):
        agent = AnalyzerAgent("NoLLM-Analyzer")
        result = await agent.think({"data": "test"})
        assert result["data_received"] is True


# ============================================================================
# AgentSystem.get_health()
# ============================================================================

class TestHealthCheck:
    def test_health_keys_present(self):
        system = AgentSystem("HealthSys")
        health = system.get_health()
        assert "status" in health
        assert "agents_healthy" in health
        assert "agents_total" in health
        assert "uptime_seconds" in health
        assert "task_success_rate" in health

    def test_healthy_status_on_fresh_system(self):
        system = AgentSystem("FreshSys")
        health = system.get_health()
        assert health["status"] == "healthy"
        assert health["agents_healthy"] == health["agents_total"]
        assert health["task_success_rate"] == 1.0

    def test_uptime_is_positive(self):
        system = AgentSystem("UptimeSys")
        health = system.get_health()
        assert health["uptime_seconds"] >= 0

    def test_degraded_when_agent_in_error(self):
        system = AgentSystem("DegradedSys")
        agent = ExecutorAgent("ErrorAgent")
        agent.status = AgentStatus.ERROR
        system.add_agent(agent)
        health = system.get_health()
        assert health["agents_healthy"] < health["agents_total"]


# ============================================================================
# AgentSystem general
# ============================================================================

class TestAgentSystem:
    def test_add_remove_agent(self):
        system = AgentSystem("ManageSys")
        agent = ExecutorAgent("Exec-manage")
        system.add_agent(agent)
        assert system.get_agent(agent.id) is agent
        system.remove_agent(agent.id)
        assert system.get_agent(agent.id) is None

    def test_system_status_keys(self):
        system = AgentSystem("StatusSys")
        status = system.get_system_status()
        assert "system_name" in status
        assert "agents" in status
        assert "metrics" in status

    def test_to_json_is_valid_json(self):
        import json
        system = AgentSystem("JSONSys")
        raw = system.to_json()
        parsed = json.loads(raw)
        assert "system_name" in parsed

    def test_factory_create_agent(self):
        agent = AgentFactory.create_agent("executor", "FactoryExec")
        assert agent is not None
        assert isinstance(agent, ExecutorAgent)

    def test_factory_unknown_type_returns_none(self):
        agent = AgentFactory.create_agent("unknown_type", "X")
        assert agent is None

    def test_factory_create_team(self):
        system = AgentFactory.create_team({"executor": 2, "analyzer": 1})
        # 1 orchestrator + 2 executors + 1 analyzer = 4 total
        assert len(system.agents) == 4


# ============================================================================
# Memory
# ============================================================================

class TestAgentMemory:
    def test_store_and_retrieve_episode(self):
        mem = AgentMemory(agent_id="test-id")
        mem.store_episode("key1", "value1")
        assert mem.retrieve("key1", "episodic") == "value1"

    def test_store_and_retrieve_semantic(self):
        mem = AgentMemory(agent_id="test-id")
        mem.store_semantic("sem_key", {"data": 42})
        assert mem.retrieve("sem_key", "semantic") == {"data": 42}

    def test_auto_retrieval_prefers_episodic(self):
        mem = AgentMemory(agent_id="test-id")
        mem.store_episode("shared", "episodic_val")
        mem.store_semantic("shared", "semantic_val")
        assert mem.retrieve("shared") == "episodic_val"

    def test_missing_key_returns_none(self):
        mem = AgentMemory(agent_id="test-id")
        assert mem.retrieve("nonexistent") is None

    def test_max_episodes_fifo_eviction(self):
        mem = AgentMemory(agent_id="test-id", max_episodes=3)
        for i in range(4):
            mem.store_episode(f"k{i}", f"v{i}")
        # First inserted key should be evicted
        assert mem.retrieve("k0", "episodic") is None
        assert mem.retrieve("k3", "episodic") == "v3"


# ============================================================================
# StructuredLogger
# ============================================================================

class TestStructuredLogger:
    def test_logger_creation(self):
        slog = StructuredLogger("test_module", agent_id="abc", agent_name="TestAgent")
        assert slog._base_extra["agent_id"] == "abc"
        assert slog._base_extra["agent_name"] == "TestAgent"

    def test_extra_includes_task_id(self):
        slog = StructuredLogger("test_module")
        extra = slog._extra(task_id="task-123")
        assert extra["structured"]["task_id"] == "task-123"

    def test_log_methods_callable(self):
        import logging
        slog = StructuredLogger("test_module")
        # Should not raise
        with patch.object(slog._logger, "info") as mock_info:
            slog.info("Test message")
            mock_info.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
