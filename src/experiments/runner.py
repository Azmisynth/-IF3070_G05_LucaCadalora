import argparse
import json
import random
from pathlib import Path
from typing import Any

from src.algorithms.genetic_algorithm import genetic_algorithm
from src.algorithms.hill_climbing import hill_climb
from src.algorithms.simulated_annealing import simulated_annealing
from src.core.initialization import generate_random_state
from src.experiments.config import load_experiment_config
from src.models.ship import Ship
from src.models.state import State
from src.models.vehicle import Vehicle


def _load_problem(path: str | Path) -> tuple[Ship, list[Vehicle]]:
    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream)
    ship = Ship.from_dict(data["ship"])
    vehicles = [Vehicle.from_dict(item) for item in data["vehicles"]]
    return ship, vehicles


def _serialize_value(value: Any) -> Any:
    if isinstance(value, State):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    return value


def _initial_state(
    ship: Ship,
    vehicles: list[Vehicle],
    seed: int,
) -> State:
    return generate_random_state(ship, vehicles, rng=random.Random(seed))


def _run_hill_climbing(
    ship: Ship,
    vehicles: list[Vehicle],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    records = []
    for run in range(1, config["runs"] + 1):
        seed = config["seed"] + run
        state = _initial_state(ship, vehicles, seed)
        result = hill_climb(
            state,
            seed=seed,
            **config["hill_climbing"],
        )
        records.append({"run": run, "seed": seed, **_serialize_value(result)})
    return records


def _run_simulated_annealing(
    ship: Ship,
    vehicles: list[Vehicle],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    records = []
    for run in range(1, config["runs"] + 1):
        seed = config["seed"] + run
        state = _initial_state(ship, vehicles, seed)
        result = simulated_annealing(
            state,
            seed=seed,
            **config["simulated_annealing"],
        )
        records.append({"run": run, "seed": seed, **_serialize_value(result)})
    return records


def _run_genetic_configuration(
    ship: Ship,
    vehicles: list[Vehicle],
    config: dict[str, Any],
    population_size: int,
    generations: int,
    experiment: str,
    seed_offset: int,
) -> list[dict[str, Any]]:
    genetic_config = config["genetic_algorithm"]
    shared_parameters = {
        key: value
        for key, value in genetic_config.items()
        if key not in {"experiment_a", "experiment_b"}
    }
    records = []
    for run in range(1, config["runs"] + 1):
        seed = config["seed"] + seed_offset + run
        state = _initial_state(ship, vehicles, seed)
        result = genetic_algorithm(
            state,
            population_size=population_size,
            generations=generations,
            seed=seed,
            **shared_parameters,
        )
        records.append(
            {
                "experiment": experiment,
                "run": run,
                "seed": seed,
                **_serialize_value(result),
            }
        )
    return records


def _run_genetic_algorithm(
    ship: Ship,
    vehicles: list[Vehicle],
    config: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    genetic_config = config["genetic_algorithm"]
    experiment_a = genetic_config["experiment_a"]
    experiment_b = genetic_config["experiment_b"]
    records_a = []
    records_b = []

    for index, generations in enumerate(experiment_a["generation_variations"]):
        records_a.extend(
            _run_genetic_configuration(
                ship,
                vehicles,
                config,
                experiment_a["fixed_population_size"],
                generations,
                "A",
                1000 + index * 100,
            )
        )

    for index, population_size in enumerate(experiment_b["population_variations"]):
        records_b.extend(
            _run_genetic_configuration(
                ship,
                vehicles,
                config,
                population_size,
                experiment_b["fixed_generations"],
                "B",
                2000 + index * 100,
            )
        )

    return {"experiment_a": records_a, "experiment_b": records_b}


def run_experiments(
    input_path: str | Path,
    output_path: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    ship, vehicles = _load_problem(input_path)
    config = load_experiment_config(config_path)
    results = {
        "input": {
            "ship": ship.to_dict(),
            "vehicles": [vehicle.to_dict() for vehicle in vehicles],
        },
        "configuration": config,
        "hill_climbing": _run_hill_climbing(ship, vehicles, config),
        "simulated_annealing": _run_simulated_annealing(ship, vehicles, config),
        "genetic_algorithm": _run_genetic_algorithm(ship, vehicles, config),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input/example.json")
    parser.add_argument("--config", default="input/experiment_config.json")
    parser.add_argument("--output", default="output/experiments.json")
    arguments = parser.parse_args()
    run_experiments(arguments.input, arguments.output, arguments.config)


if __name__ == "__main__":
    main()
