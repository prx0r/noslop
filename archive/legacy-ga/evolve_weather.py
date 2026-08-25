"""Weather evolution: simple loop version (no async hang)."""
import asyncio
import json
import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "repos/cogym"))

from cogym_kernel.kernel.runner import AsyncRunner, ExecutorRegistry
from cogym_kernel.kernel.contracts import CandidateArtifact

from worldpacks.weather.world import WeatherWorld
from worldpacks.weather.policy import FormatPolicy, SEARCH_SPACE, TEMPLATES
from worldpacks.weather.executor import WeatherExecutor
from cogym_kernel.evo.recipes import propose_children, EvolutionContext, sample_config, mutate_config


async def evaluate(runner, configs, suite):
    results = []
    for cfg in configs:
        policy = FormatPolicy(cfg)
        scores = []
        for instance_id, seed in suite:
            world = WeatherWorld(query_type="check")
            rec = await runner.run_episode(world, policy, instance_id=instance_id,
                                           seed=seed, max_steps=5)
            scores.append(rec.metrics.get("overlap") or 0)
        avg = sum(scores) / len(scores) if scores else 0
        results.append({"config": cfg, "avg_overlap": avg, "scores": scores})
    return results


async def main():
    runner = AsyncRunner(ExecutorRegistry({"deterministic": WeatherExecutor()}))
    suite = [(f"city_{i}", i) for i in range(15)]
    rng = random.Random(42)

    # Generate initial population from all templates
    population = []
    for key in TEMPLATES:
        population.append({
            "template_key": key,
            "include_period": True,
            "capitalize_first": True,
        })

    generations = 50
    population_size = 8
    elite_k = 3

    print(f"Generations: {generations}, Population: {population_size}, Seeds: {len(population)}")
    print()

    for gen in range(generations):
        # Evaluate current population
        evaluated = await evaluate(runner, population, suite)
        evaluated.sort(key=lambda e: e["avg_overlap"], reverse=True)

        # Print generation stats
        scores = [e["avg_overlap"] for e in evaluated]
        best = max(scores)
        avg = sum(scores) / len(scores)
        passing = sum(1 for s in scores if s >= 0.50)
        print(f"Gen {gen:2d}: avg={avg:.3f} best={best:.3f} pass={passing}/{len(scores)} "
              f"top={evaluated[0]['config']['template_key']}")

        if gen == generations - 1:
            break

        # Select elites
        elites = [e["config"] for e in evaluated[:elite_k]]

        # Generate children via mutation
        children = []
        for i in range(population_size - elite_k):
            parent = elites[i % len(elites)]
            child = mutate_config(parent, SEARCH_SPACE, rng, rate=0.4)
            children.append(child)

        population = elites + children

    # Final results
    print("\n=== WINNERS ===")
    for i, e in enumerate(evaluated[:3]):
        print(f"\n#{i+1}: {e['config']['template_key']}")
        print(f"  Avg overlap: {e['avg_overlap']:.3f}")
        # Test on a specific city
        policy = FormatPolicy(e["config"])
        world = WeatherWorld(query_type="check")
        rec = await runner.run_episode(world, policy, instance_id="london", seed=42, max_steps=5)
        print(f"  London test: {rec.metrics.get('overlap'):.3f}")
        print(f"  Run ID: {rec.run_id}")

    # Save
    results = {
        "winners": [e["config"] for e in evaluated[:3]],
        "generations": generations,
        "scores": {e["config"]["template_key"]: e["avg_overlap"] for e in evaluated},
    }
    with open("weather_evolution_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to weather_evolution_results.json")


if __name__ == "__main__":
    asyncio.run(main())
