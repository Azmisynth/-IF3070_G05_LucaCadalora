from dataclasses import replace
import json
from pathlib import Path
import random
import tempfile
import unittest

from src.algorithms.genetic_algorithm import _build_valid_child, _crossover, genetic_algorithm
from src.algorithms.hill_climbing import VALID_VARIANTS, hill_climb
from src.algorithms.simulated_annealing import simulated_annealing
from src.core.constraints import has_overlap, is_inside_boundary, is_valid_state, within_capacity
from src.core.initialization import create_outside_state, generate_random_state
from src.core.neighbor import generate_neighbors, move_vehicle, random_neighbor, rotate_vehicle, swap_vehicles, transfer_and_load, unload_vehicle
from src.core.objective import calculate_objective, ship_summaries, with_updated_metrics
from src.io.problem import load_problem
from src.models.fleet_state import FleetState, state_key
from src.models.ship import Ship
from src.models.state import Orientation, Placement, State
from src.models.vehicle import Vehicle


class FleetTest(unittest.TestCase):
    def setUp(self):
        self.ships = {"K1": Ship(2, 2, 5), "K2": Ship(3, 2, 10)}
        self.first = Vehicle("A", 2, 2, 10, 5, 1)
        self.second = Vehicle("B", 2, 2, 20, 5, 3)
        self.third = Vehicle("C", 1, 1, 6, 2, 0)
        self.state = with_updated_metrics(FleetState(self.ships, [
            Placement(self.first, True, 0, 0, ship_id="K1"),
            Placement.outside(self.second), Placement.outside(self.third),
        ]))

    def assert_vehicles_preserved(self, state):
        self.assertEqual(state.vehicles, self.state.vehicles)
        self.assertEqual(len({p.vehicle.id for p in state.placements}), len(state.placements))

    def test_coordinates_on_different_ships_do_not_overlap(self):
        state = self.state.replace_placement(1, Placement(self.second, True, 0, 0, ship_id="K2"))
        self.assertTrue(is_valid_state(state))
        self.assertFalse(has_overlap(state))
        self.assertEqual(calculate_objective(state), 30)
        self.assertEqual(calculate_objective(state, "fee_per_area"), 7.5)
        self.assertEqual(calculate_objective(state, "fee_with_eta"), 10)

    def test_each_ship_enforces_its_own_boundary_and_capacity(self):
        state = FleetState(self.ships, [
            Placement(self.first, True, 0, 0, ship_id="K1"),
            Placement(self.third, True, 0, 0, ship_id="K1"),
        ], total_weight=0)
        self.assertFalse(within_capacity(state))
        self.assertTrue(has_overlap(state))
        self.assertFalse(is_valid_state(state))
        overflow = self.state.replace_placement(0, replace(self.state.placements[0], x=1))
        self.assertFalse(is_inside_boundary(overflow))
        self.assertTrue(within_capacity(self.state))

    def test_invalid_assignments_and_duplicate_ids_are_rejected(self):
        invalid = [
            [Placement(self.first, True, 0, 0)],
            [Placement(self.first, True, 0, 0, ship_id="missing")],
            [Placement(self.first, False, ship_id="K1")],
            [Placement(self.first, False, 0, 0)],
            [Placement(self.first, True, .5, 0, ship_id="K1")],
            [Placement.outside(self.first), Placement.outside(self.first)],
        ]
        for placements in invalid:
            with self.subTest(placements=placements), self.assertRaises(ValueError):
                FleetState(self.ships, placements)
        with self.assertRaises(ValueError):
            FleetState({}, [])
        with self.assertRaises(ValueError):
            self.state.replace_placements(self.state.placements[:-1])

    def test_transfer_rotation_swap_and_unload_are_atomic(self):
        before = self.state.to_dict()
        transferred = move_vehicle(self.state, "A", 1, 0, ship_id="K2")
        self.assertIsNotNone(transferred)
        self.assertEqual(transferred.placements[0].ship_id, "K2")
        self.assertEqual(transferred.objective_value, 10)
        self.assertIsNone(move_vehicle(self.state, "B", 0, 0, ship_id="K1"))
        self.assertIsNone(move_vehicle(self.state, "A", 0, 0, ship_id="missing"))
        loaded = move_vehicle(self.state, "B", 0, 0, ship_id="K2")
        swapped = swap_vehicles(loaded, "A", "B")
        self.assertEqual([p.ship_id for p in swapped.placements[:2]], ["K2", "K1"])
        rotated = rotate_vehicle(transferred, "A")
        self.assertEqual(rotated.placements[0].ship_id, "K2")
        self.assertEqual(rotated.placements[0].orientation, Orientation.VERTICAL)
        unloaded = unload_vehicle(transferred, "A")
        self.assertEqual(unloaded.objective_value, 0)
        self.assertEqual(unloaded.placements[0], Placement.outside(self.first))
        for state in (transferred, swapped, rotated, unloaded):
            self.assertTrue(is_valid_state(state))
            self.assert_vehicles_preserved(state)
        self.assertEqual(self.state.to_dict(), before)

    def test_cross_ship_swap_rejects_destination_capacity_violation(self):
        heavy = Vehicle("H", 1, 1, 12, 6, 0)
        state = FleetState(self.ships, [
            Placement(self.third, True, 0, 0, ship_id="K1"),
            Placement(heavy, True, 0, 0, ship_id="K2"),
        ])
        self.assertIsNone(swap_vehicles(state, "C", "H"))
        self.assertTrue(is_valid_state(state))

    def test_transfer_and_load_improves_revenue_in_one_step(self):
        candidate = transfer_and_load(self.state, "A", "B", "K2", 0, 0)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.objective_value, 30)
        self.assertEqual(candidate.placements[0].ship_id, "K2")
        self.assertEqual(candidate.placements[1].ship_id, "K1")
        self.assertTrue(is_valid_state(candidate))
        self.assertIsNone(transfer_and_load(self.state, "A", "B", "K2", 2, 1))
        self.assertEqual(self.state.objective_value, 10)

    def test_neighbors_preserve_vehicles_and_include_ship_identity(self):
        neighbors = generate_neighbors(self.state)
        keys = [state_key(state) for state in neighbors]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertNotIn(state_key(self.state), keys)
        self.assertTrue(any(state.objective_value == 30 for state in neighbors))
        for state in neighbors:
            self.assertTrue(is_valid_state(state))
            self.assert_vehicles_preserved(state)
        transferred = move_vehicle(self.state, "A", 0, 0, ship_id="K2")
        self.assertNotEqual(state_key(self.state), state_key(transferred))

    def test_random_initialization_is_valid_reproducible_and_handles_unfit_vehicles(self):
        ships = {**self.ships, "K0": Ship(10, 10, 0)}
        oversized = Vehicle("X", 8, 8, 50, 1, 0)
        vehicles = [*self.state.vehicles, oversized]
        for seed in range(10):
            first = generate_random_state(ships, vehicles, rng=random.Random(seed))
            second = generate_random_state(ships, vehicles, rng=random.Random(seed))
            self.assertTrue(is_valid_state(first))
            self.assertEqual(first.to_dict(), second.to_dict())
            self.assertFalse(first.placements[-1].inside)
            self.assertEqual(first.vehicles, vehicles)
        state = generate_random_state({"K": Ship(1, 2, 5)}, [self.first], rng=random.Random(0))
        self.assertFalse(state.placements[0].inside)

    def test_initialization_tries_other_ships_and_orientations(self):
        vehicle = Vehicle("R", 3, 2, 10, 4, 0)
        state = generate_random_state(
            {"small": Ship(1, 1, 10), "fit": Ship(2, 3, 4)}, [vehicle],
            inclusion_probability=1, rng=random.Random(2),
        )
        self.assertEqual(state.placements[0].ship_id, "fit")
        self.assertEqual(state.placements[0].orientation, Orientation.VERTICAL)
        self.assertTrue(is_valid_state(state))

    def test_random_neighbors_preserve_validity(self):
        state = self.state
        generator = random.Random(12)
        for _ in range(100):
            candidate = random_neighbor(state, generator)
            if candidate is not None:
                self.assertNotEqual(state_key(state), state_key(candidate))
                self.assertTrue(is_valid_state(candidate))
                self.assert_vehicles_preserved(candidate)
                state = candidate

    def test_one_ship_fleet_has_same_constraints_and_scores(self):
        legacy = State(self.ships["K1"], [Placement(self.first, True, 0, 0), Placement.outside(self.second)])
        fleet = FleetState({"K1": legacy.ship}, [replace(p, ship_id="K1" if p.inside else None) for p in legacy.placements])
        self.assertEqual(is_valid_state(legacy), is_valid_state(fleet))
        for mode in ("shipping_fee", "fee_per_area", "fee_with_eta"):
            self.assertEqual(calculate_objective(legacy, mode), calculate_objective(fleet, mode))
        self.assertNotIn("shipId", legacy.to_dict()["placements"][0])

    def test_ship_summaries_use_local_capacity_and_loaded_vehicles(self):
        summaries = ship_summaries(self.state)
        self.assertEqual(summaries[0]["loaded_ids"], ["A"])
        self.assertEqual(summaries[0]["weight_utilization"], 1)
        self.assertEqual(summaries[0]["area_utilization"], 1)
        self.assertEqual(summaries[1]["loaded_count"], 0)
        self.assertEqual(sum(row["shipping_fee"] for row in summaries), 10)

    def test_all_algorithms_support_fleets_and_reproduce_results(self):
        calls = [(hill_climb, {"variant": variant, "max_iterations": 4, "max_restart": 1})
                 for variant in sorted(VALID_VARIANTS)]
        calls.extend([
            (simulated_annealing, {"max_iterations": 30}),
            (genetic_algorithm, {"population_size": 6, "generations": 5}),
        ])
        for algorithm, parameters in calls:
            with self.subTest(algorithm=algorithm.__name__, parameters=parameters):
                first = algorithm(self.state, seed=4, **parameters)
                second = algorithm(self.state, seed=4, **parameters)
                self.assertEqual(first["history"], second["history"])
                self.assertEqual(first["final_state"].to_dict(), second["final_state"].to_dict())
                self.assertTrue(is_valid_state(first["final_state"]))
                self.assert_vehicles_preserved(first["final_state"])
                self.assertGreaterEqual(first["best_objective"], self.state.objective_value)
                if algorithm is hill_climb:
                    self.assertEqual(first["best_objective"], 36)

    def test_genetic_repair_relocates_conflicts_without_losing_vehicles(self):
        genes = [
            Placement(self.first, True, 0, 0, ship_id="K1"),
            Placement(self.second, True, 0, 0, ship_id="K1"),
            Placement(self.third, True, 0, 0, ship_id="K2"),
        ]
        child = _build_valid_child(self.state, genes, "shipping_fee", random.Random(3))
        self.assertTrue(is_valid_state(child))
        self.assert_vehicles_preserved(child)
        self.assertEqual(child.placements[1].ship_id, "K2")
        other = move_vehicle(self.state, "B", 0, 0, ship_id="K2")
        for child in _crossover(self.state, other, random.Random(2), 1, "shipping_fee"):
            self.assertTrue(is_valid_state(child))
            self.assert_vehicles_preserved(child)

    def test_loader_rejects_invalid_fleet_inputs(self):
        original = {"ships": [{"id": "K", "width": 2, "length": 2, "maxCapacity": 5}],
                    "vehicles": [self.first.to_dict()]}
        invalid = [
            {**original, "ships": []},
            {**original, "ship": original["ships"][0]},
            {**original, "ships": original["ships"] * 2},
            {**original, "ships": [{**original["ships"][0], "id": ""}]},
            {**original, "ships": [{**original["ships"][0], "width": 1.5}]},
            {**original, "ships": [{**original["ships"][0], "maxCapacity": float("inf")}]},
            {**original, "vehicles": original["vehicles"] * 2},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(json.dumps(original))
            ships, vehicles = load_problem(path)
            self.assertEqual(ships, {"K": Ship(2, 2, 5)})
            self.assertEqual(vehicles, [self.first])
            for data in invalid:
                with self.subTest(data=data):
                    path.write_text(json.dumps(data))
                    with self.assertRaises(ValueError):
                        load_problem(path)


if __name__ == "__main__":
    unittest.main()
