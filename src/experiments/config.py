import json
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_EXPERIMENT_CONFIG: dict[str, Any] = {
    "runs": 3,
    "seed": 2026,
    "hill_climbing": {
        "variant": "steepest",
        "max_iterations": 1000,
        "max_sideways": 20,
        "max_restart": 10,
    },
    "simulated_annealing": {
        "initial_temperature": 100.0,
        "cooling_rate": 0.995,
        "minimum_temperature": 0.01,
        "max_iterations": 1000,
        "neighbor_attempts": 100,
        "stuck_threshold": 25,
    },
    "genetic_algorithm": {
        "crossover_rate": 0.8,
        "mutation_rate": 0.2,
        "elitism_count": 2,
        "tournament_size": 3,
        "mutation_attempts": 50,
        "experiment_a": {
            "fixed_population_size": 30,
            "generation_variations": [50, 100, 200],
        },
        "experiment_b": {
            "fixed_generations": 100,
            "population_variations": [10, 30, 50],
        },
    },
}


def _merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_experiment_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return deepcopy(DEFAULT_EXPERIMENT_CONFIG)
    with Path(path).open(encoding="utf-8") as stream:
        override = json.load(stream)
    return _merge_config(DEFAULT_EXPERIMENT_CONFIG, override)
