"""Tests for noslop classifier worldpack."""
import asyncio
import pytest
from cogym_kernel.kernel.runner import AsyncRunner, ExecutorRegistry
from cogym_kernel.kernel.contracts import CandidateArtifact
from .world import NoslopWorld
from .policy import ClassifierPolicy
from .executor import ClassifierExecutor


@pytest.fixture
def runner():
    return AsyncRunner(ExecutorRegistry({"deterministic": ClassifierExecutor()}))


@pytest.mark.asyncio
async def test_determinism(runner):
    world = NoslopWorld()
    policy = ClassifierPolicy({"narr_weight": 1.0})
    cand = CandidateArtifact(kind="classifier", version="1", config={"narr_weight": 1.0})
    r1 = await runner.run_episode(world, policy, instance_id="test", seed=0,
                                  candidate=cand, max_steps=5)
    r2 = await runner.run_episode(world, policy, instance_id="test", seed=0,
                                  candidate=cand, max_steps=5)
    assert r1.run_id == r2.run_id


@pytest.mark.asyncio
async def test_classifies(runner):
    world = NoslopWorld()
    policy = ClassifierPolicy()
    r = await runner.run_episode(world, policy, instance_id="test", seed=0,
                                 max_steps=5)
    assert r.metrics.get("accuracy") is not None


@pytest.mark.asyncio
async def test_scoring(runner):
    world = NoslopWorld()
    policy = ClassifierPolicy()
    r = await runner.run_episode(world, policy, instance_id="test", seed=0,
                                 max_steps=5)
    acc = r.metrics.get("accuracy")
    assert acc is not None
    assert 0.0 <= acc <= 1.0
