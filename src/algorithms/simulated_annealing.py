import math
import random
from time import perf_counter
from typing import Any

from src.core.constraints import is_valid_state
from src.core.neighbor import random_neighbor
from src.core.objective import with_updated_metrics
from src.models.fleet_state import SearchState


def simulated_annealing(
    state: SearchState,
    initial_temperature: float = 100.0,
    cooling_rate: float = 0.995,
    minimum_temperature: float = 0.01,
    max_iterations: int = 1000,
    neighbor_attempts: int = 100,
    stuck_threshold: int = 25,
    objective_mode: str = "shipping_fee",
    seed: int | None = None,
) -> dict[str, Any]:
    if not is_valid_state(state):
        raise ValueError("Initial state must satisfy all constraints")
    if initial_temperature <= 0 or minimum_temperature <= 0:
        raise ValueError("Temperatures must be positive")
    if not 0 < cooling_rate < 1:
        raise ValueError("cooling_rate must be between 0 and 1")
    if max_iterations < 0 or neighbor_attempts <= 0 or stuck_threshold <= 0:
        raise ValueError("Iteration settings must be positive")

    generator = random.Random(seed)
    initial = with_updated_metrics(state, objective_mode)
    current = initial
    best = initial
    temperature = initial_temperature
    iterations_without_improvement = 0
    stuck_count = 0
    accepted_moves = 0
    started_at = perf_counter()
    history: list[dict[str, float | int | bool]] = [
        {
            "iteration": 0,
            "objective": current.objective_value,
            "best_objective": best.objective_value,
            "temperature": temperature,
            "acceptance_probability": 1.0,
            "accepted": True,
            "stuck": False,
        }
    ]

    iteration = 0
    while iteration < max_iterations and temperature > minimum_temperature:
        candidate = random_neighbor(
            current,
            generator,
            objective_mode,
            neighbor_attempts,
        )
        if candidate is None:
            break

        delta = candidate.objective_value - current.objective_value
        acceptance_probability = 1.0 if delta >= 0 else math.exp(delta / temperature)
        accepted = delta >= 0 or generator.random() < acceptance_probability
        if accepted:
            current = candidate
            accepted_moves += 1

        if current.objective_value > best.objective_value:
            best = current
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1

        stuck = iterations_without_improvement == stuck_threshold
        if stuck:
            stuck_count += 1
            iterations_without_improvement = 0

        iteration += 1
        temperature *= cooling_rate
        history.append(
            {
                "iteration": iteration,
                "objective": current.objective_value,
                "best_objective": best.objective_value,
                "temperature": temperature,
                "acceptance_probability": acceptance_probability,
                "accepted": accepted,
                "stuck": stuck,
            }
        )

    return {
        "initial_state": initial,
        "final_state": best,
        "best_objective": best.objective_value,
        "iterations": iteration,
        "duration": perf_counter() - started_at,
        "history": history,
        "accepted_moves": accepted_moves,
        "stuck_count": stuck_count,
        "initial_temperature": initial_temperature,
        "final_temperature": temperature,
        "cooling_rate": cooling_rate,
        "objective_mode": objective_mode,
    }
