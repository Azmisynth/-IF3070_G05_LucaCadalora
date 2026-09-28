import random
from time import perf_counter
from typing import Any

from src.core.constraints import is_valid_state
from src.core.initialization import generate_random_state
from src.core.neighbor import generate_neighbors
from src.core.objective import with_updated_metrics
from src.models.state import State


VALID_VARIANTS = {
    "first_improvement",
    "steepest",
    "stochastic",
    "sideways",
    "random_restart",
}


def _state_key(state: State) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            placement.vehicle.id,
            placement.inside,
            placement.x,
            placement.y,
            placement.orientation.value,
        )
        for placement in state.placements
    )


def _pick_neighbor(
    current: State,
    neighbors: list[State],
    variant: str,
    generator: random.Random,
    sideways_used: int,
    max_sideways: int,
    seen: set[tuple[tuple[object, ...], ...]],
) -> State | None:
    better = [
        neighbor
        for neighbor in neighbors
        if neighbor.objective_value > current.objective_value
    ]

    if variant == "first_improvement":
        return better[0] if better else None
    if variant == "steepest":
        return max(better, key=lambda state: state.objective_value, default=None)
    if variant == "stochastic":
        return generator.choice(better) if better else None

    if better:
        return max(better, key=lambda state: state.objective_value)
    if sideways_used >= max_sideways:
        return None
    sideways = [
        neighbor
        for neighbor in neighbors
        if neighbor.objective_value == current.objective_value
        and _state_key(neighbor) not in seen
    ]
    return generator.choice(sideways) if sideways else None


def _single_hill_climb(
    initial_state: State,
    variant: str,
    objective_mode: str,
    max_iterations: int,
    max_sideways: int,
    generator: random.Random,
) -> dict[str, Any]:
    initial = with_updated_metrics(initial_state, objective_mode)
    current = initial
    history = [{"iteration": 0, "objective": current.objective_value}]
    seen = {_state_key(current)}
    sideways_used = 0
    iterations = 0

    while iterations < max_iterations:
        neighbors = generate_neighbors(current, objective_mode)
        selected = _pick_neighbor(
            current,
            neighbors,
            variant,
            generator,
            sideways_used,
            max_sideways,
            seen,
        )
        if selected is None:
            break

        if selected.objective_value == current.objective_value:
            sideways_used += 1
        else:
            sideways_used = 0

        current = selected
        seen.add(_state_key(current))
        iterations += 1
        history.append({"iteration": iterations, "objective": current.objective_value})

    return {
        "initial_state": initial,
        "final_state": current,
        "best_objective": current.objective_value,
        "iterations": iterations,
        "history": history,
        "sideways_moves": sum(
            first["objective"] == second["objective"]
            for first, second in zip(history, history[1:])
        ),
    }


def _random_restart(
    initial_state: State,
    objective_mode: str,
    max_iterations: int,
    max_restart: int,
    generator: random.Random,
) -> dict[str, Any]:
    runs = [
        _single_hill_climb(
            initial_state,
            "steepest",
            objective_mode,
            max_iterations,
            0,
            generator,
        )
    ]

    for _ in range(max_restart):
        restart_state = generate_random_state(
            initial_state.ship,
            initial_state.vehicles,
            objective_mode=objective_mode,
            rng=generator,
        )
        runs.append(
            _single_hill_climb(
                restart_state,
                "steepest",
                objective_mode,
                max_iterations,
                0,
                generator,
            )
        )

    best_run = max(runs, key=lambda run: run["best_objective"])
    history = []
    global_iteration = 0
    for restart, run in enumerate(runs):
        for entry in run["history"]:
            history.append(
                {
                    "iteration": global_iteration,
                    "restart": restart,
                    "objective": entry["objective"],
                }
            )
            global_iteration += 1

    return {
        "initial_state": initial_state,
        "final_state": best_run["final_state"],
        "best_objective": best_run["best_objective"],
        "iterations": sum(run["iterations"] for run in runs),
        "history": history,
        "restarts": max_restart,
        "iterations_per_restart": [run["iterations"] for run in runs],
        "maximum_restart": max_restart,
    }


def hill_climb(
    state: State,
    variant: str = "steepest",
    max_iterations: int = 1000,
    max_sideways: int = 20,
    max_restart: int = 10,
    objective_mode: str = "shipping_fee",
    seed: int | None = None,
) -> dict[str, Any]:
    if not is_valid_state(state):
        raise ValueError("Initial state must satisfy all constraints")
    if max_iterations < 0 or max_sideways < 0 or max_restart < 0:
        raise ValueError("Iteration limits cannot be negative")

    if variant not in VALID_VARIANTS:
        choices = ", ".join(sorted(VALID_VARIANTS))
        raise ValueError(f"Unknown Hill Climbing variant: {variant}. Choose from {choices}")

    generator = random.Random(seed)
    started_at = perf_counter()
    if variant == "random_restart":
        result = _random_restart(
            state,
            objective_mode,
            max_iterations,
            max_restart,
            generator,
        )
    else:
        result = _single_hill_climb(
            state,
            variant,
            objective_mode,
            max_iterations,
            max_sideways,
            generator,
        )
        if variant == "sideways":
            result["maximum_sideways"] = max_sideways

    result["duration"] = perf_counter() - started_at
    result["variant"] = variant
    result["objective_mode"] = objective_mode
    return result
