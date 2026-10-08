import unittest

from src.algorithms.genetic_algorithm import genetic_algorithm
from src.core.constraints import is_valid_state
from src.core.initialization import create_outside_state
from src.models.ship import Ship
from src.models.vehicle import Vehicle


class GeneticAlgorithmTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = create_outside_state(
            Ship(width=6, length=5, max_capacity=16),
            [
                Vehicle("A", 2, 2, 10, 4, 1),
                Vehicle("B", 2, 1, 8, 3, 3),
                Vehicle("C", 1, 1, 6, 2, 0),
                Vehicle("D", 3, 2, 12, 6, 2),
                Vehicle("E", 2, 3, 14, 7, 1),
            ],
        )

    def test_result_contains_fitness_history(self) -> None:
        result = genetic_algorithm(
            self.state,
            population_size=12,
            generations=20,
            seed=13,
        )

        required_keys = {
            "initial_state",
            "final_state",
            "best_objective",
            "generations",
            "duration",
            "history",
            "population_size",
            "crossover_rate",
            "mutation_rate",
            "elitism_count",
            "tournament_size",
        }
        self.assertEqual(required_keys - result.keys(), set())
        self.assertTrue(is_valid_state(result["final_state"]))
        self.assertEqual(len(result["history"]), 21)
        self.assertEqual(result["history"][-1]["generation"], 20)
        self.assertEqual(
            result["best_objective"],
            max(entry["max_fitness"] for entry in result["history"]),
        )

    def test_elitism_keeps_maximum_fitness(self) -> None:
        result = genetic_algorithm(
            self.state,
            population_size=10,
            generations=15,
            elitism_count=2,
            seed=21,
        )

        maximum_fitness = [entry["max_fitness"] for entry in result["history"]]
        self.assertTrue(
            all(
                current <= following
                for current, following in zip(maximum_fitness, maximum_fitness[1:])
            )
        )

    def test_seed_makes_evolution_reproducible(self) -> None:
        first = genetic_algorithm(
            self.state,
            population_size=9,
            generations=12,
            seed=8,
        )
        second = genetic_algorithm(
            self.state,
            population_size=9,
            generations=12,
            seed=8,
        )

        self.assertEqual(first["history"], second["history"])
        self.assertEqual(first["best_objective"], second["best_objective"])
        self.assertEqual(
            first["final_state"].to_dict(),
            second["final_state"].to_dict(),
        )

    def test_zero_generations_returns_valid_initial_population_result(self) -> None:
        result = genetic_algorithm(
            self.state,
            population_size=6,
            generations=0,
            seed=5,
        )

        self.assertEqual(len(result["history"]), 1)
        self.assertEqual(result["generations"], 0)
        self.assertTrue(is_valid_state(result["final_state"]))

    def test_rejects_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            genetic_algorithm(self.state, population_size=1)
        with self.assertRaises(ValueError):
            genetic_algorithm(self.state, mutation_rate=1.1)
        with self.assertRaises(ValueError):
            genetic_algorithm(self.state, population_size=4, elitism_count=4)
        with self.assertRaises(ValueError):
            genetic_algorithm(self.state, population_size=4, tournament_size=5)


if __name__ == "__main__":
    unittest.main()
