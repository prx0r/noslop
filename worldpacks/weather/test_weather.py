"""Determinism tests for weather worldpack.

Same (worldpack, instance_id, seed, candidate) => same run_id.
"""
import asyncio
import pytest

from cogym_kernel.kernel.runner import AsyncRunner, ExecutorRegistry
from cogym_kernel.kernel.contracts import CandidateArtifact

from .world import WeatherWorld
from .policy import FormatPolicy
from .executor import WeatherExecutor


@pytest.fixture
def runner():
    return AsyncRunner(ExecutorRegistry({"deterministic": WeatherExecutor()}))


@pytest.mark.asyncio
async def test_determinism(runner):
    """Same inputs produce same run_id."""
    world = WeatherWorld(query_type="check")
    policy = FormatPolicy({"template_key": "gt_concise_city_first"})
    cand = CandidateArtifact(kind="weather_policy", version="1",
                             config={"template_key": "gt_concise_city_first"})

    r1 = await runner.run_episode(world, policy, instance_id="london", seed=42,
                                  candidate=cand, max_steps=5)
    r2 = await runner.run_episode(world, policy, instance_id="london", seed=42,
                                  candidate=cand, max_steps=5)
    assert r1.run_id == r2.run_id, f"Non-deterministic: {r1.run_id} != {r2.run_id}"


@pytest.mark.asyncio
async def test_formats_response(runner):
    """Policy produces a formatted response."""
    world = WeatherWorld(query_type="check")
    policy = FormatPolicy({"template_key": "gt_concise_city_first"})

    r = await runner.run_episode(world, policy, instance_id="paris", seed=7,
                                 max_steps=5)
    # Should have produced a response with overlap > 0
    overlap = r.metrics.get("overlap")
    assert overlap is not None, "No overlap metric"
    assert overlap > 0.0, f"Zero overlap: {r.to_json()}"


@pytest.mark.asyncio
async def test_different_templates(runner):
    """Different configs produce different run_ids."""
    world1 = WeatherWorld(query_type="check")
    world2 = WeatherWorld(query_type="check")
    p1 = FormatPolicy({"template_key": "gt_concise_city_first"})
    p2 = FormatPolicy({"template_key": "gt_sentence_current"})

    r1 = await runner.run_episode(world1, p1, instance_id="london", seed=42,
                                  max_steps=5)
    r2 = await runner.run_episode(world2, p2, instance_id="london", seed=42,
                                  max_steps=5)
    assert r1.run_id != r2.run_id


@pytest.mark.asyncio
async def test_scoring(runner):
    """Scoring returns valid metrics."""
    world = WeatherWorld(query_type="check")
    policy = FormatPolicy({"template_key": "gt_sentence_current"})

    r = await runner.run_episode(world, policy, instance_id="tokyo", seed=10,
                                 max_steps=5)
    overlap = r.metrics.get("overlap")
    assert overlap is not None
    assert 0.0 <= overlap <= 1.0, f"Overlap out of range: {overlap}"
