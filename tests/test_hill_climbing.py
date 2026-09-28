import unittest

from src.algorithms.hill_climbing import hill_climb
from src.core.constraints import is_valid_state
from src.core.initialization import create_outside_state
from src.models.ship import Ship
from src.models.vehicle import Vehicle


class HillClimbingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = create_outside_state(
            Ship(width=4, length=3, max_capacity=8),
            [
                Vehicle("A", 2, 2, 10, 4, 1),
                Vehicle("B", 2, 1, 8, 3, 3),
                Vehicle("C", 1, 1, 6, 2, 0),
            ],
        )

    def test_all_variants_return_shared_contract(self) -> None:
        variants = [
            "first_improvement",
            "steepest",
            "stochastic",
            "sideways",
            "random_restart",
        ]

        for variant in variants:
            with self.subTest(variant=variant):
                result = hill_climb(
                    self.state,
                    variant=variant,
                    max_iterations=20,
                    max_sideways=3,
                    max_restart=2,
                    seed=7,
                )
                self.assertEqual(
                    {
                        "initial_state",
                        "final_state",
                        "best_objective",
                        "iterations",
                        "duration",
                        "history",
                    }
                    - result.keys(),
                    set(),
                )
                self.assertTrue(is_valid_state(result["final_state"]))
                self.assertGreaterEqual(result["best_objective"], 0)
                self.assertGreaterEqual(result["duration"], 0)

    def test_result_is_reproducible_with_seed(self) -> None:
        first = hill_climb(self.state, variant="random_restart", max_restart=2, seed=31)
        second = hill_climb(self.state, variant="random_restart", max_restart=2, seed=31)

        self.assertEqual(first["best_objective"], second["best_objective"])
        self.assertEqual(first["history"], second["history"])
        self.assertEqual(first["final_state"].to_dict(), second["final_state"].to_dict())

    def test_sideways_and_random_restart_metadata(self) -> None:
        sideways = hill_climb(
            self.state,
            variant="sideways",
            max_sideways=4,
            seed=5,
        )
        restarted = hill_climb(
            self.state,
            variant="random_restart",
            max_restart=3,
            seed=5,
        )

        self.assertEqual(sideways["maximum_sideways"], 4)
        self.assertLessEqual(sideways["sideways_moves"], 4)
        self.assertEqual(restarted["restarts"], 3)
        self.assertEqual(len(restarted["iterations_per_restart"]), 4)

    def test_variant_must_use_a_fixed_choice(self) -> None:
        with self.assertRaises(ValueError):
            hill_climb(self.state, variant="steepest_ascent")


if __name__ == "__main__":
    unittest.main()
