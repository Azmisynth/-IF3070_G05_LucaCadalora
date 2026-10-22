import random
from time import perf_counter
from typing import Any

from src.core.constraints import is_valid_state
from src.core.initialization import create_outside_state, generate_random_state, place_randomly
from src.core.neighbor import random_neighbor
from src.core.objective import with_updated_metrics
from src.models.state import Placement
from src.models.fleet_state import FleetState, SearchState, ship_configuration


def _build_valid_child(
    parent: SearchState,
    genes: list[Placement],
    objective_mode: str,
    generator: random.Random,
) -> SearchState:
    child = create_outside_state(ship_configuration(parent), parent.vehicles, objective_mode)
    for index, gene in enumerate(genes):
        if not gene.inside:
            continue
        candidate = child.replace_placement(index, gene)
        if is_valid_state(candidate):
            child = candidate
        elif isinstance(child, FleetState):
            child = place_randomly(child, index, generator, objective_mode, max_attempts=20)
    return with_updated_metrics(child, objective_mode)


def _crossover(
    first: SearchState,
    second: SearchState,
    generator: random.Random,
    crossover_rate: float,
    objective_mode: str,
) -> tuple[SearchState, SearchState]:
    if len(first.placements) < 2 or generator.random() >= crossover_rate:
        return (
            with_updated_metrics(first, objective_mode),
            with_updated_metrics(second, objective_mode),
        )
    point = generator.randrange(1, len(first.placements))
    first_genes = first.placements[:point] + second.placements[point:]
    second_genes = second.placements[:point] + first.placements[point:]
    return (
        _build_valid_child(first, first_genes, objective_mode, generator),
        _build_valid_child(second, second_genes, objective_mode, generator),
    )


def _mutate(
    state: SearchState,
    generator: random.Random,
    mutation_rate: float,
    mutation_attempts: int,
    objective_mode: str,
) -> SearchState:
    if generator.random() >= mutation_rate:
        return state
    candidate = random_neighbor(state, generator, objective_mode, mutation_attempts)
    return candidate if candidate is not None else state


def _select_parent(
    population: list[SearchState],
    tournament_size: int,
    generator: random.Random,
) -> SearchState:
    candidates = generator.sample(population, tournament_size)
    return max(candidates, key=lambda state: state.objective_value)


def _generation_summary(
    population: list[SearchState],
    generation: int,
) -> dict[str, float | int]:
    fitness_values = [state.objective_value for state in population]
    return {
        "generation": generation,
        "max_fitness": max(fitness_values),
        "average_fitness": sum(fitness_values) / len(fitness_values),
        "min_fitness": min(fitness_values),
    }


def genetic_algorithm(
    state: SearchState,
    population_size: int = 30,
    generations: int = 100,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.2,
    elitism_count: int = 2,
    tournament_size: int = 3,
    mutation_attempts: int = 50,
    objective_mode: str = "shipping_fee",
    seed: int | None = None,
) -> dict[str, Any]:
    if not is_valid_state(state):
        raise ValueError("Initial state must satisfy all constraints")
    if population_size < 2:
        raise ValueError("population_size must be at least 2")
    if generations < 0:
        raise ValueError("generations cannot be negative")
    if not 0 <= crossover_rate <= 1 or not 0 <= mutation_rate <= 1:
        raise ValueError("Crossover and mutation rates must be between 0 and 1")
    if not 0 <= elitism_count < population_size:
        raise ValueError("elitism_count must be smaller than population_size")
    if not 1 <= tournament_size <= population_size:
        raise ValueError("tournament_size must be within the population size")
    if mutation_attempts <= 0:
        raise ValueError("mutation_attempts must be positive")

    generator = random.Random(seed)
    initial = with_updated_metrics(state, objective_mode)
    population = [initial]
    for _ in range(population_size - 1):
        population.append(
            generate_random_state(
                ship_configuration(state),
                state.vehicles,
                objective_mode=objective_mode,
                rng=generator,
            )
        )

    best = max(population, key=lambda candidate: candidate.objective_value)
    history = [_generation_summary(population, 0)]
    started_at = perf_counter()

    for generation in range(1, generations + 1):
        ranked = sorted(
            population,
            key=lambda candidate: candidate.objective_value,
            reverse=True,
        )
        next_population = ranked[:elitism_count]

        while len(next_population) < population_size:
            first_parent = _select_parent(population, tournament_size, generator)
            second_parent = _select_parent(population, tournament_size, generator)
            children = _crossover(
                first_parent,
                second_parent,
                generator,
                crossover_rate,
                objective_mode,
            )
            for child in children:
                mutated = _mutate(
                    child,
                    generator,
                    mutation_rate,
                    mutation_attempts,
                    objective_mode,
                )
                next_population.append(mutated)
                if len(next_population) == population_size:
                    break

        population = next_population
        generation_best = max(
            population,
            key=lambda candidate: candidate.objective_value,
        )
        if generation_best.objective_value > best.objective_value:
            best = generation_best
        history.append(_generation_summary(population, generation))

    return {
        "initial_state": initial,
        "final_state": best,
        "best_objective": best.objective_value,
        "generations": generations,
        "duration": perf_counter() - started_at,
        "history": history,
        "population_size": population_size,
        "crossover_rate": crossover_rate,
        "mutation_rate": mutation_rate,
        "elitism_count": elitism_count,
        "tournament_size": tournament_size,
        "objective_mode": objective_mode,
    }
