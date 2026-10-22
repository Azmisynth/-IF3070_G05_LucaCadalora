import random
import unittest

from src.core.constraints import has_overlap, is_inside_boundary, is_valid_state, within_capacity
from src.core.initialization import create_outside_state, generate_random_state
from src.core.neighbor import generate_neighbors, move_vehicle, rotate_vehicle, swap_vehicles
from src.core.objective import calculate_objective, with_updated_metrics
from src.models.ship import Ship
from src.models.state import Orientation, Placement, State
from src.models.vehicle import Vehicle


class CoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ship = Ship(width=4, length=3, max_capacity=8)
        self.first = Vehicle("A", 2, 2, 10, 4, 1)
        self.second = Vehicle("B", 2, 1, 8, 3, 3)
        self.third = Vehicle("C", 1, 1, 6, 2, 0)

    def test_boundary_overlap_and_capacity(self) -> None:
        valid = State(
            self.ship,
            [
                Placement(self.first, True, 0, 0),
                Placement(self.second, True, 2, 0),
            ],
        )
        overlap = State(
            self.ship,
            [
                Placement(self.first, True, 0, 0),
                Placement(self.second, True, 1, 1),
            ],
        )
        outside_boundary = State(
            self.ship,
            [Placement(self.first, True, 3, 2)],
        )
        over_capacity = State(
            Ship(4, 3, 6),
            [
                Placement(self.first, True, 0, 0),
                Placement(self.second, True, 2, 0),
            ],
        )

        self.assertTrue(is_valid_state(valid))
        self.assertTrue(has_overlap(overlap))
        self.assertFalse(is_inside_boundary(outside_boundary))
        self.assertFalse(within_capacity(over_capacity))

    def test_objective_modes(self) -> None:
        state = State(
            self.ship,
            [
                Placement(self.first, True, 0, 0),
                Placement(self.second, True, 2, 0),
                Placement.outside(self.third),
            ],
        )

        self.assertEqual(calculate_objective(state), 18)
        self.assertEqual(calculate_objective(state, "fee_per_area"), 6.5)
        self.assertEqual(calculate_objective(state, "fee_with_eta"), 7)

    def test_random_initial_state_is_valid_and_reproducible(self) -> None:
        vehicles = [self.first, self.second, self.third]
        first_state = generate_random_state(
            self.ship,
            vehicles,
            rng=random.Random(12),
        )
        second_state = generate_random_state(
            self.ship,
            vehicles,
            rng=random.Random(12),
        )

        self.assertTrue(is_valid_state(first_state))
        self.assertEqual(first_state.to_dict(), second_state.to_dict())
        self.assertEqual(
            first_state.total_weight,
            sum(item.vehicle.weight for item in first_state.inside_placements),
        )

    def test_neighbor_operations_keep_state_valid(self) -> None:
        state = with_updated_metrics(
            State(
                self.ship,
                [
                    Placement(self.first, True, 0, 0),
                    Placement(self.second, True, 2, 0),
                    Placement.outside(self.third),
                ],
            )
        )

        moved = move_vehicle(state, "B", 2, 2)
        rotated = rotate_vehicle(state, "B")
        swapped = swap_vehicles(state, "A", "C")

        self.assertIsNotNone(moved)
        self.assertIsNotNone(rotated)
        self.assertIsNotNone(swapped)
        self.assertTrue(is_valid_state(moved))
        self.assertTrue(is_valid_state(rotated))
        self.assertTrue(is_valid_state(swapped))

    def test_generated_neighbors_are_unique_and_valid(self) -> None:
        state = create_outside_state(
            self.ship,
            [self.first, self.second, self.third],
        )
        neighbors = generate_neighbors(state)
        serialized = [str(neighbor.to_dict()["placements"]) for neighbor in neighbors]

        self.assertTrue(neighbors)
        self.assertTrue(all(is_valid_state(neighbor) for neighbor in neighbors))
        self.assertEqual(len(serialized), len(set(serialized)))
        self.assertTrue(any(neighbor.objective_value > 0 for neighbor in neighbors))


if __name__ == "__main__":
    unittest.main()
