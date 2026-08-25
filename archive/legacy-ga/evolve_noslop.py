"""NoSlop classifier evolution: evolve optimal detection weights via cogym.

Run: python3 evolve_noslop.py
"""
import asyncio
import json
import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "repos/cogym"))

from cogym_kernel.kernel.runner import AsyncRunner, ExecutorRegistry
from cogym_kernel.kernel.contracts import CandidateArtifact

from worldpacks.noslop.world import NoslopWorld
from worldpacks.noslop.policy import ClassifierPolicy, SEARCH_SPACE
from worldpacks.noslop.executor import ClassifierExecutor
from cogym_kernel.evo.recipes import mutate_config, sample_config


async def evaluate(runner, configs, dataset_size):
    results = []
    for cfg in configs:
        policy = ClassifierPolicy(cfg)
        correct = 0
        total = 0
        for seed in range(dataset_size):
            world = NoslopWorld()
            rec = await runner.run_episode(world, policy, instance_id=f"test_{seed}",
                                           seed=seed, max_steps=5)
            acc = rec.metrics.get("accuracy") or 0
            correct += acc
            total += 1
        avg_acc = correct / total if total > 0 else 0
        results.append({"config": cfg, "accuracy": avg_acc})
    return results


async def main():
    runner = AsyncRunner(ExecutorRegistry({"deterministic": ClassifierExecutor()}))

    # Load dataset size
    with open("data/test_dataset.json") as f:
        dataset = json.load(f)
    dataset_size = len(dataset["entries"])

    rng = random.Random(42)
    generations = 100
    population_size = 12
    elite_k = 4

    # Generate initial population
    population = []
    for _ in range(population_size):
        population.append(sample_config(SEARCH_SPACE, rng))

    print(f"Generations: {generations}, Population: {population_size}, "
          f"Dataset: {dataset_size} entries")
    print()

    best_ever = 0
    best_config = None

    for gen in range(generations):
        evaluated = await evaluate(runner, population, dataset_size)
        evaluated.sort(key=lambda e: e["accuracy"], reverse=True)

        scores = [e["accuracy"] for e in evaluated]
        best = max(scores)
        avg = sum(scores) / len(scores)

        if best > best_ever:
            best_ever = best
            best_config = evaluated[0]["config"]

        # Print every 10 generations
        if gen % 10 == 0 or gen == generations - 1:
            print(f"Gen {gen:3d}: avg={avg:.3f} best={best:.3f} best_ever={best_ever:.3f} "
                  f"top={evaluated[0]['config']}")

        if gen == generations - 1:
            break

        # Select elites
        elites = [e["config"] for e in evaluated[:elite_k]]

        # Generate children via mutation
        children = []
        for i in range(population_size - elite_k):
            parent = elites[i % len(elites)]
            child = mutate_config(parent, SEARCH_SPACE, rng, rate=0.3)
            children.append(child)

        population = elites + children

    # Final results
    print(f"\n=== WINNER ===")
    print(f"Best accuracy: {best_ever:.3f}")
    print(f"Config: {json.dumps(best_config, indent=2)}")

    # Test on each entry
    policy = ClassifierPolicy(best_config)
    print(f"\n=== PER-ENTRY RESULTS ===")
    for i, entry in enumerate(dataset["entries"]):
        world = NoslopWorld()
        rec = await runner.run_episode(world, policy, instance_id=f"test_{i}",
                                       seed=i, max_steps=5)
        acc = rec.metrics.get("accuracy") or 0
        predicted = 1 if acc == 1.0 else 0
        match = "OK" if predicted == entry["label"] else "MISS"
        print(f"  {match} {entry['id']}: pred={predicted} true={entry['label']}")

    # Save results
    results = {
        "winner_config": best_config,
        "best_accuracy": best_ever,
        "generations": generations,
        "dataset_size": dataset_size,
    }
    with open("noslop_evolution_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to noslop_evolution_results.json")


if __name__ == "__main__":
    asyncio.run(main())
