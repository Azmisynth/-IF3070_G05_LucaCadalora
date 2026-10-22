import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.main import main, load_problem
from src.models.ship import Ship
from src.models.vehicle import Vehicle
from src.models.state import Orientation, Placement, State
from src.core.objective import with_updated_metrics
from src.visualization.grid import draw_state
from src.visualization.export import restore_state, serialize


class VisualizationIntegrationTest(unittest.TestCase):
    def test_rotated_rectangle_and_state_roundtrip(self):
        vehicle = Vehicle("A", 2, 3, 100, 40, 2)
        state = with_updated_metrics(State(Ship(6, 8, 100),
                    [Placement(vehicle, True, 1, 2, Orientation.VERTICAL)]))
        fig = draw_state(state)
        try:
            rect = next(p for p in fig.axes[0].patches if isinstance(p, Rectangle))
            self.assertEqual((rect.get_x(), rect.get_y(), rect.get_width(), rect.get_height()), (1, 2, 3, 2))
            restored = restore_state(serialize(state))
            self.assertEqual(restored.to_dict(), state.to_dict())
        finally:
            plt.close(fig)

    def test_cli_real_algorithms_and_serialized_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            for algorithm, params, expected in [
                ("hc", ["--max-iterations", "2"], "objective_plot.png"),
                ("sa", ["--max-iterations", "5"], "probability_plot.png"),
                ("ga", ["--population", "4", "--generations", "2"], "fitness_max_average_plot.png")]:
                destination = Path(directory) / algorithm
                with contextlib.redirect_stdout(io.StringIO()):
                    code = main(["--algorithm", algorithm, "--output", str(destination), *params])
                self.assertEqual(code, 0)
                for name in ("initial_state.png", "final_state.png", "result.json", "history.csv", expected):
                    self.assertGreater((destination / name).stat().st_size, 0)
                data = json.loads((destination / "result.json").read_text())
                from src.core.constraints import is_valid_state
                self.assertTrue(is_valid_state(restore_state(data["final_state"])))
                self.assertEqual(data["seed"], 2026)

    def test_input_rejects_fractional_dimensions_and_duplicate_ids(self):
        original = json.loads(Path("input/example.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            original["ship"]["width"] = 2.5
            path.write_text(json.dumps(original))
            with self.assertRaises(ValueError):
                load_problem(path)
            original["ship"]["width"] = 10
            original["vehicles"].append(original["vehicles"][0])
            path.write_text(json.dumps(original))
            with self.assertRaises(ValueError):
                load_problem(path)


if __name__ == "__main__":
    unittest.main()
