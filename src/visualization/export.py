import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

from src.core.constraints import is_valid_state
from src.core.objective import calculate_objective, ship_summaries
from src.models.ship import Ship
from src.models.vehicle import Vehicle
from src.models.state import State, Placement
from src.models.fleet_state import FleetState
from src.io.problem import parse_ships
from src.visualization.grid import draw_state
from src.visualization.plot import plot_objective, plot_sa_probability, plot_ga_fitness


def serialize(value):
    if isinstance(value, (State, FleetState)):
        return value.to_dict()
    if isinstance(value, dict):
        return {k: serialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serialize(v) for v in value]
    return value


def restore_state(data):
    fleet = "ships" in data
    if fleet and "ship" in data:
        raise ValueError("State tidak boleh memiliki ship dan ships bersamaan")
    placements = [
        Placement(Vehicle.from_dict(p["vehicle"]), inside=p["inside"], x=p["x"], y=p["y"],
                  orientation=p["orientation"], ship_id=p.get("shipId")) for p in data["placements"]]
    metrics = {"objective_value": data["objectiveValue"], "total_weight": data["totalWeight"]}
    if fleet:
        return FleetState(parse_ships(data["ships"]), placements, **metrics)
    return State(Ship.from_dict(data["ship"]), placements, **metrics)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_result(result, destination, algorithm):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    mode = result.get("objective_mode", "shipping_fee")
    result = dict(result)
    summaries = []
    for key, title in [("initial_state", "Initial state"), ("final_state", "Final / best state")]:
        state = result[key]
        if isinstance(state, dict):
            state = restore_state(state)
        if not is_valid_state(state):
            raise ValueError(f"{key} tidak valid; output dihentikan")
        if isinstance(state, FleetState):
            result.update(problem_type="multi_ship", ship_count=len(state.ships))
            result[f"{key}_ships"] = ship_summaries(state, mode)
            for summary in result[f"{key}_ships"]:
                summaries.append({"stage": key, **summary, "loaded_ids": ", ".join(summary["loaded_ids"])})
        fig = draw_state(state, f"{algorithm.upper()} - {title}", destination / f"{key}.png", mode)
        plt.close(fig)
    history = result["history"]
    fleet = result.get("problem_type") == "multi_ship"
    if algorithm == "ga":
        plt.close(plot_ga_fitness(history, destination / "fitness_max_average_plot.png",
                                 title="Genetic Algorithm - fitness armada" if fleet else "Genetic Algorithm - fitness populasi"))
    else:
        plt.close(plot_objective(history, destination / "objective_plot.png",
                                 title="Objective armada vs iterasi" if fleet else "Objective vs iterasi"))
    if algorithm == "sa":
        plt.close(plot_sa_probability(history, destination / "probability_plot.png"))
    write_csv(destination / "history.csv", history)
    if summaries:
        write_csv(destination / "ships.csv", summaries)
    with (destination / "result.json").open("w", encoding="utf-8") as stream:
        json.dump(serialize(result), stream, indent=2, allow_nan=False)


def export_experiments(results, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    groups = [("hc", results["hill_climbing"]), ("sa", results["simulated_annealing"]),
              ("ga", results["genetic_algorithm"]["experiment_a"]),
              ("ga", results["genetic_algorithm"]["experiment_b"])]
    for algorithm, records in groups:
        for r in records:
            experiment = r.get("experiment", "-")
            tag = (f"ga_{experiment}_p{r['population_size']}_g{r['generations']}_run{r['run']}"
                   if algorithm == "ga" else f"{algorithm}_run{r['run']}")
            export_result(r, destination / tag, algorithm)
            final = restore_state(r["final_state"]) if isinstance(r["final_state"], dict) else r["final_state"]
            rows.append({"algorithm": algorithm, "experiment": experiment, "run": r["run"],
                         "seed": r["seed"], "variant": r.get("variant", ""),
                         "objective_mode": r.get("objective_mode", "shipping_fee"),
                         "objective": r["best_objective"], "shipping_fee": calculate_objective(final),
                         "iterations": r.get("iterations", r.get("generations")),
                         "population_size": r.get("population_size", ""),
                         "generations": r.get("generations", ""), "duration": r["duration"],
                         "stuck_count": r.get("stuck_count", ""),
                         "valid": is_valid_state(final)})
            if isinstance(final, FleetState):
                rows[-1].update(ship_count=len(final.ships), loaded_count=len(final.inside_placements),
                                total_weight=final.total_weight)
    write_csv(destination / "results.csv", rows)
    for experiment, parameter in [("A", "generations"), ("B", "population_size")]:
        grouped = defaultdict(list)
        for row in rows:
            if row["algorithm"] == "ga" and row["experiment"] == experiment:
                grouped[row[parameter]].append(row["objective"])
        if not grouped:
            continue
        x = sorted(grouped)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.errorbar(x, [mean(grouped[v]) for v in x],
                    yerr=[stdev(grouped[v]) if len(grouped[v]) > 1 else 0 for v in x],
                    marker="o", capsize=5, label="Rata-rata +/- simpangan baku sampel")
        for v in x:
            ax.scatter([v] * len(grouped[v]), grouped[v], alpha=.5, s=18)
        ax.set(xlabel="Jumlah generasi" if experiment == "A" else "Ukuran populasi",
               ylabel="Objective terbaik per run", title=f"GA - eksperimen {experiment}")
        ax.set_xticks(x)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        name = "ga_iteration_comparison.png" if experiment == "A" else "ga_population_comparison.png"
        fig.savefig(destination / name, dpi=150)
        plt.close(fig)
    return rows
