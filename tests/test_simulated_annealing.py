import unittest

from src.algorithms.simulated_annealing import simulated_annealing
from src.core.constraints import is_valid_state
from src.core.initialization import create_outside_state
from src.models.ship import Ship
from src.models.vehicle import Vehicle


class SimulatedAnnealingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = create_outside_state(
            Ship(width=5, length=4, max_capacity=12),
            [
                Vehicle("A", 2, 2, 10, 4, 1),
                Vehicle("B", 2, 1, 8, 3, 3),
                Vehicle("C", 1, 1, 6, 2, 0),
                Vehicle("D", 3, 2, 12, 6, 2),
            ],
        )

    def test_result_contains_experiment_data(self) -> None:
        result = simulated_annealing(
            self.state,
            initial_temperature=20,
            cooling_rate=0.95,
            max_iterations=80,
            seed=17,
        )

        required_keys = {
            "initial_state",
            "final_state",
            "best_objective",
            "iterations",
            "duration",
            "history",
            "accepted_moves",
            "stuck_count",
            "initial_temperature",
            "final_temperature",
            "cooling_rate",
        }
        self.assertEqual(required_keys - result.keys(), set())
        self.assertTrue(is_valid_state(result["final_state"]))
        self.assertGreaterEqual(
            result["best_objective"],
            result["initial_state"].objective_value,
        )
        self.assertEqual(len(result["history"]), result["iterations"] + 1)

    def test_temperature_and_probability_history(self) -> None:
        result = simulated_annealing(
            self.state,
            initial_temperature=15,
            cooling_rate=0.9,
            minimum_temperature=0.1,
            max_iterations=30,
            seed=4,
        )

        temperatures = [entry["temperature"] for entry in result["history"]]
        probabilities = [
            entry["acceptance_probability"] for entry in result["history"]
        ]
        self.assertTrue(
            all(current > following for current, following in zip(temperatures, temperatures[1:]))
        )
        self.assertTrue(all(0 <= probability <= 1 for probability in probabilities))
        self.assertEqual(
            result["best_objective"],
            max(entry["best_objective"] for entry in result["history"]),
        )

    def test_seed_makes_search_reproducible(self) -> None:
        first = simulated_annealing(self.state, max_iterations=60, seed=29)
        second = simulated_annealing(self.state, max_iterations=60, seed=29)

        self.assertEqual(first["history"], second["history"])
        self.assertEqual(first["best_objective"], second["best_objective"])
        self.assertEqual(
            first["final_state"].to_dict(),
            second["final_state"].to_dict(),
        )

    def test_rejects_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            simulated_annealing(self.state, initial_temperature=0)
        with self.assertRaises(ValueError):
            simulated_annealing(self.state, cooling_rate=1)
        with self.assertRaises(ValueError):
            simulated_annealing(self.state, neighbor_attempts=0)


if __name__ == "__main__":
    unittest.main()
