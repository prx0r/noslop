"""NoSlop classifier evolution v2: expanded features + expanded dataset.

Run: python3 evolve_noslop_v2.py
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

    with open("data/test_dataset.json") as f:
        dataset = json.load(f)
    dataset_size = len(dataset["entries"])

    rng = random.Random(42)
    generations = 150
    population_size = 20
    elite_k = 6

    # Expanded search space for all features
    search_space = {
        "narr_weight": [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0],
        "neg_weight": [0.2, 0.3, 0.5, 0.8, 1.0, 1.5],
        "three_list_weight": [0.2, 0.3, 0.5, 0.8, 1.0, 1.5],
        "cliche_weight": [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
        "pattern_threshold": [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
        "burstiness_threshold": [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6],
        "punct_threshold": [0.005, 0.01, 0.02, 0.03, 0.04, 0.05],
        "ld_threshold": [0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
        "ngram_threshold": [0.2, 0.3, 0.5, 0.8, 1.0, 1.5],
        "fw_threshold": [0.30, 0.35, 0.40, 0.45, 0.50, 0.55],
        "the_threshold": [0.2, 0.25, 0.30, 0.35, 0.40, 0.50],
        "abstract_threshold": [0.4, 0.5, 0.6, 0.7, 0.8],
        "commitment_threshold": [0.05, 0.1, 0.15, 0.2, 0.3],
        "ai_threshold": [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    }

    population = [sample_config(search_space, rng) for _ in range(population_size)]

    print(f"Generations: {generations}, Population: {population_size}, "
          f"Dataset: {dataset_size} entries, Elite: {elite_k}")
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

        if gen % 10 == 0 or gen == generations - 1:
            print(f"Gen {gen:3d}: avg={avg:.3f} best={best:.3f} best_ever={best_ever:.3f}")

        if gen == generations - 1:
            break

        elites = [e["config"] for e in evaluated[:elite_k]]
        children = []
        for i in range(population_size - elite_k):
            parent = elites[i % len(elites)]
            child = mutate_config(parent, search_space, rng, rate=0.3)
            children.append(child)
        population = elites + children

    print(f"\n=== WINNER ===")
    print(f"Best accuracy: {best_ever:.3f}")
    print(f"Config: {json.dumps(best_config, indent=2)}")

    # Per-entry results
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

    # Feature importance summary
    print(f"\n=== FEATURE IMPORTANCE ===")
    print(f"  NARR weight: {best_config.get('narr_weight', 1.0)}")
    print(f"  NEG weight: {best_config.get('neg_weight', 1.0)}")
    print(f"  3LIST weight: {best_config.get('three_list_weight', 1.0)}")
    print(f"  CLICHE weight: {best_config.get('cliche_weight', 1.0)}")
    print(f"  LD threshold: {best_config.get('ld_threshold', 0.3)}")
    print(f"  N-gram threshold: {best_config.get('ngram_threshold', 0.5)}")
    print(f"  FW threshold: {best_config.get('fw_threshold', 0.45)}")
    print(f"  AI threshold: {best_config.get('ai_threshold', 0.5)}")

    results = {
        "winner_config": best_config,
        "best_accuracy": best_ever,
        "generations": generations,
        "dataset_size": dataset_size,
        "search_space": search_space,
    }
    with open("noslop_evolution_v2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to noslop_evolution_v2_results.json")


if __name__ == "__main__":
    asyncio.run(main())
