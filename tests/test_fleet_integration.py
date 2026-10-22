import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.core.constraints import is_valid_state
from src.core.initialization import generate_random_state
from src.experiments.runner import run_experiments
from src.io.problem import load_problem
from src.main import main
from src.models.fleet_state import FleetState
from src.visualization.export import export_experiments, restore_state, serialize
from src.visualization.grid import draw_state


class FleetIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.input_path = Path("input/multi_ship_example.json")

    def test_fleet_roundtrip_and_grid_preserve_assignments(self):
        ships, vehicles = load_problem(self.input_path)
        import random
        state = generate_random_state(ships, vehicles, inclusion_probability=1, rng=random.Random(6))
        restored = restore_state(serialize(state))
        self.assertEqual(state.to_dict(), restored.to_dict())
        self.assertTrue(is_valid_state(restored))
        fig = draw_state(restored)
        try:
            self.assertEqual(len(fig.axes), len(ships) + 1)
            for ax, ship_id in zip(fig.axes, ships):
                self.assertEqual(ax.get_title(), ship_id)
                rectangles = [patch for patch in ax.patches if isinstance(patch, Rectangle)]
                placements = [p for p in state.inside_placements if p.ship_id == ship_id]
                self.assertEqual(len(rectangles), len(placements))
                for rectangle, placement in zip(rectangles, placements):
                    self.assertEqual(
                        (rectangle.get_x(), rectangle.get_y(), rectangle.get_width(), rectangle.get_height()),
                        (placement.x, placement.y, placement.placed_width, placement.placed_length),
                    )
            info = fig.axes[-1].texts[0].get_text()
            self.assertEqual(info.count("Outside ("), 1)
            self.assertIn("Objective armada", info)
        finally:
            plt.close(fig)

    def test_cli_algorithms_export_valid_fleet_results(self):
        with tempfile.TemporaryDirectory() as directory:
            for algorithm, parameters in [
                ("hc", ["--max-iterations", "2"]),
                ("sa", ["--max-iterations", "10"]),
                ("ga", ["--population", "4", "--generations", "2"]),
            ]:
                with self.subTest(algorithm=algorithm):
                    destination = Path(directory) / algorithm
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        code = main(["--input", str(self.input_path), "--algorithm", algorithm,
                                     "--output", str(destination), "--objective", "fee_with_eta", *parameters])
                    self.assertEqual(code, 0)
                    self.assertIn("K1:", output.getvalue())
                    self.assertIn("K2:", output.getvalue())
                    for name in ("initial_state.png", "final_state.png", "ships.csv", "result.json", "history.csv"):
                        self.assertGreater((destination / name).stat().st_size, 0)
                    data = json.loads((destination / "result.json").read_text())
                    self.assertEqual(data["problem_type"], "multi_ship")
                    self.assertEqual(data["ship_count"], 2)
                    self.assertEqual(data["objective_mode"], "fee_with_eta")
                    final = restore_state(data["final_state"])
                    self.assertIsInstance(final, FleetState)
                    self.assertTrue(is_valid_state(final))
                    self.assertEqual(len(final.vehicles), 7)
                    self.assertAlmostEqual(
                        sum(summary["objective"] for summary in data["final_state_ships"]),
                        data["best_objective"],
                    )
                    with (destination / "ships.csv").open(newline="") as stream:
                        self.assertEqual(len(list(csv.DictReader(stream))), 4)

    def test_experiments_and_restored_exports_support_fleet_input(self):
        config = {
            "runs": 1,
            "hill_climbing": {"max_iterations": 1},
            "simulated_annealing": {"max_iterations": 3},
            "genetic_algorithm": {
                "experiment_a": {"fixed_population_size": 4, "generation_variations": [1]},
                "experiment_b": {"fixed_generations": 1, "population_variations": [4]},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            result_path = root / "experiments.json"
            result = run_experiments(self.input_path, result_path, config_path)
            self.assertIn("ships", result["input"])
            self.assertNotIn("ship", result["input"])
            restored = json.loads(result_path.read_text())
            rows = export_experiments(restored, root / "rendered")
            self.assertEqual(len(rows), 4)
            for row in rows:
                self.assertTrue(row["valid"])
                self.assertEqual(row["ship_count"], 2)
                self.assertLessEqual(row["loaded_count"], 6)
            for path in (root / "rendered").glob("*/result.json"):
                data = json.loads(path.read_text())
                self.assertTrue(is_valid_state(restore_state(data["final_state"])))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(["--render-experiments", str(result_path), "--output", str(root / "cli_rendered")])
            self.assertEqual(code, 0)
            self.assertTrue((root / "cli_rendered" / "results.csv").exists())

    def test_restoration_rejects_unknown_and_duplicate_ship_ids(self):
        ships, vehicles = load_problem(self.input_path)
        state = FleetState(ships, [])
        data = state.to_dict()
        data["ships"].append(data["ships"][0])
        with self.assertRaises(ValueError):
            restore_state(data)


if __name__ == "__main__":
    unittest.main()
