import argparse
import json
import math
from pathlib import Path
import random
import sys
from datetime import datetime

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.algorithms.hill_climbing import hill_climb, VALID_VARIANTS
from src.algorithms.simulated_annealing import simulated_annealing
from src.algorithms.genetic_algorithm import genetic_algorithm
from src.core.initialization import generate_random_state
from src.core.constraints import is_valid_state
from src.core.objective import OBJECTIVE_MODES, calculate_objective
from src.models.ship import Ship
from src.models.vehicle import Vehicle
from src.visualization.export import export_result, export_experiments
from src.experiments.runner import run_experiments

ROOT = Path(__file__).resolve().parents[1]


def load_problem(path):
    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or not isinstance(data.get("ship"), dict) or not isinstance(data.get("vehicles"), list):
        raise ValueError("Input memerlukan object ship dan array vehicles")

    for item in [data["ship"], *data["vehicles"]]:
        if not isinstance(item, dict):
            raise ValueError("Setiap kendaraan harus berupa object")
        for key, value in item.items():
            if key == "id":
                continue
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"{key} harus berupa angka finite")
        for key in ("width", "length"):
            if type(item.get(key)) is not int or item[key] <= 0:
                raise ValueError(f"{key} harus integer positif")
    ship = Ship.from_dict(data["ship"])
    vehicles = [Vehicle.from_dict(v) for v in data["vehicles"]]
    if not vehicles:
        raise ValueError("Minimal satu kendaraan diperlukan")
    if len({v.id for v in vehicles}) != len(vehicles):
        raise ValueError("ID kendaraan harus unik")
    return ship, vehicles


def print_result(result, algorithm):
    print(f"\nALGORITMA: {algorithm.upper()} | mode: {result['objective_mode']}")
    for key, label in [("initial_state", "INITIAL"), ("final_state", "FINAL / BEST")]:
        state = result[key]
        print(f"\n{label}")
        print("Inside :", ", ".join(p.vehicle.id for p in state.inside_placements) or "-")
        print("Outside:", ", ".join(p.vehicle.id for p in state.outside_placements) or "-")
        print(f"Objective: {state.objective_value:g} | ShippingFee: {calculate_objective(state):g}")
        print(f"Berat: {state.total_weight:g} / {state.ship.max_capacity:g} | Valid: {is_valid_state(state)}")
    print(f"\nIterasi/generasi: {result.get('iterations', result.get('generations'))}")
    print(f"Durasi algoritma (tanpa render): {result['duration']:.6f} detik")
    for key in ("variant", "maximum_sideways", "sideways_moves", "restarts", "iterations_per_restart",
                "maximum_restart", "stuck_count", "accepted_moves", "population_size"):
        if key in result:
            print(f"{key}: {result[key]}")


def build_parser():
    parser = argparse.ArgumentParser(description="Optimasi muatan kapal - LucaCadalora")
    parser.add_argument("--input", type=Path, default=ROOT / "input/example.json")
    parser.add_argument("--output", type=Path, help="Folder baru / kosong untuk hasil")
    parser.add_argument("--algorithm", choices=["hc", "sa", "ga"])
    parser.add_argument("--variant", choices=sorted(VALID_VARIANTS), default="steepest")
    parser.add_argument("--objective", choices=list(OBJECTIVE_MODES), default="shipping_fee")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--max-sideways", type=int, default=20)
    parser.add_argument("--max-restart", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=100)
    parser.add_argument("--cooling-rate", type=float, default=.995)
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--params", type=Path, help="JSON object berisi parameter tambahan/override algoritma")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--experiments", action="store_true", help="Jalankan runner lalu ekspor grafik dan CSV")
    group.add_argument("--render-experiments", type=Path, help="Ekspor JSON eksperimen yang sudah tersedia")
    parser.add_argument("--config", type=Path, default=ROOT / "input/experiment_config.json")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.experiments and (args.objective != "shipping_fee" or args.params):
            raise ValueError("Eksperimen memakai --config; --params/--objective hanya untuk run tunggal")

        if not args.render_experiments:
            ship, vehicles = load_problem(args.input)
        algorithm = args.algorithm
        if not args.experiments and not args.render_experiments and algorithm is None:
            print("1. Hill Climbing\n2. Simulated Annealing\n3. Genetic Algorithm")
            algorithm = {"1": "hc", "2": "sa", "3": "ga"}.get(input("Pilih algoritma [1-3]: ").strip())
            if algorithm is None:
                raise ValueError("Pilihan harus 1, 2, atau 3")
            if algorithm == "hc":
                variants = ["first_improvement", "steepest", "stochastic", "sideways", "random_restart"]
                for i, variant in enumerate(variants, 1):
                    print(f"{i}. {variant}")
                value = input("Pilih varian [Enter = steepest]: ").strip()
                if value:
                    if value not in {"1", "2", "3", "4", "5"}:
                        raise ValueError("Pilihan varian harus 1-5")
                    args.variant = variants[int(value) - 1]
        tag = "experiments" if args.experiments or args.render_experiments else algorithm
        destination = args.output or ROOT / "output" / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{tag}"
        if destination.exists() and any(destination.iterdir()):
            raise ValueError("Folder output tidak kosong; pilih folder baru agar hasil lama tidak tertimpa")
        if args.experiments:
            results = run_experiments(args.input, destination / "experiments.json", args.config)
            rows = export_experiments(results, destination)
            print(f"Eksperimen selesai: {len(rows)} run. CSV dan gambar tersimpan.")
        elif args.render_experiments:
            with args.render_experiments.open(encoding="utf-8") as stream:
                results = json.load(stream)
            rows = export_experiments(results, destination)
            print(f"Ekspor selesai: {len(rows)} run.")
        else:
            params = {"objective_mode": args.objective, "seed": args.seed}
            if algorithm == "hc":
                params.update(variant=args.variant, max_iterations=args.max_iterations,
                              max_sideways=args.max_sideways, max_restart=args.max_restart)
            elif algorithm == "sa":
                params.update(max_iterations=args.max_iterations, initial_temperature=args.temperature,
                              cooling_rate=args.cooling_rate)
            else:
                params.update(population_size=args.population, generations=args.generations)
            if args.params:
                with args.params.open(encoding="utf-8") as stream:
                    extra = json.load(stream)
                if not isinstance(extra, dict):
                    raise ValueError("File params harus berisi JSON object")
                params.update(extra)
            state = generate_random_state(ship, vehicles, objective_mode=params["objective_mode"],
                                          rng=random.Random(params["seed"]))
            function = {"hc": hill_climb, "sa": simulated_annealing, "ga": genetic_algorithm}[algorithm]
            result = function(state, **params)

            result.update(algorithm=algorithm, seed=params["seed"], parameters=params,
                          input_file=str(args.input.resolve()))
            export_result(result, destination, algorithm)
            print_result(result, algorithm)
        print(f"\nFolder output: {destination.resolve()}")
        return 0
    except (OSError, ValueError, TypeError, KeyError, EOFError) as error:
        print(f"Gagal: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
