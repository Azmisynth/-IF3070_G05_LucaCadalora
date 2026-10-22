from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from src.core.constraints import is_valid_state
from src.core.objective import calculate_objective, calculate_total_weight, ship_summaries
from src.models.fleet_state import FleetState


def draw_state(state, title="State", save_path=None, objective_mode="shipping_fee"):
    if isinstance(state, FleetState):
        return draw_fleet_state(state, title, save_path, objective_mode)

    fig, (ax, info) = plt.subplots(1, 2, figsize=(12, 8),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    colors = {p.vehicle.id: plt.get_cmap("tab20")(i % 20)
              for i, p in enumerate(state.placements)}
    ax.set(xlim=(0, state.ship.width), ylim=(0, state.ship.length),
           xlabel="x / lebar (grid)", ylabel="y / panjang (grid)", title=title)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=15, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=20, integer=True))
    ax.grid(alpha=.25)
    ax.set_axisbelow(True)
    for p in state.inside_placements:
        ax.add_patch(Rectangle((p.x, p.y), p.placed_width, p.placed_length,
                              facecolor=colors[p.vehicle.id], edgecolor="#253248", lw=1.2))
        label = f"{p.vehicle.id}\n{'H' if p.orientation.value == 'horizontal' else 'V'}"
        ax.text(p.x + p.placed_width / 2, p.y + p.placed_length / 2, label,
                ha="center", va="center", fontsize=8, clip_on=True)
    info.axis("off")
    inside = ", ".join(p.vehicle.id for p in state.inside_placements) or "-"
    outside = ", ".join(p.vehicle.id for p in state.outside_placements) or "-"
    blocks = ["RINGKASAN MUATAN", f"Mode: {objective_mode}",
              f"Objective: {calculate_objective(state, objective_mode):g}",
              f"Total ShippingFee: {calculate_objective(state):g}",
              f"Berat: {calculate_total_weight(state):g} / {state.ship.max_capacity:g}",
              f"Valid: {'Ya' if is_valid_state(state) else 'Tidak'}", "",
              f"Inside ({len(state.inside_placements)}): {inside}", "",
              f"Outside ({len(state.outside_placements)}): {outside}", "",
              "H: width x length asli", "V: width dan length ditukar", "Titik (0,0): kiri bawah"]
    lines = [line for block in blocks for line in (textwrap.wrap(block, 44) or [""])]

    fig.set_figheight(max(8, len(lines) * .24))
    info.text(0, 1, "\n".join(lines), va="top", fontsize=10, linespacing=1.5,
              transform=info.transAxes)
    fig.tight_layout()
    if save_path is not None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=150, bbox_inches="tight")
    return fig


def draw_fleet_state(state, title="State", save_path=None, objective_mode="shipping_fee"):
    summaries = ship_summaries(state, objective_mode)
    colors = {placement.vehicle.id: plt.get_cmap("tab20")(index % 20)
              for index, placement in enumerate(state.placements)}
    blocks = [title, "RINGKASAN ARMADA", f"Mode: {objective_mode}",
              f"Objective armada: {calculate_objective(state, objective_mode):g}",
              f"Total ShippingFee: {calculate_objective(state):g}",
              f"Total berat: {calculate_total_weight(state):g}",
              f"Valid: {'Ya' if is_valid_state(state) else 'Tidak'}", ""]
    for summary in summaries:
        loaded = ", ".join(summary["loaded_ids"]) or "-"
        blocks.extend([
            summary["ship_id"],
            f"Inside ({summary['loaded_count']}): {loaded}",
            f"Berat: {summary['weight']:g} / {summary['capacity']:g}",
            f"Area: {summary['occupied_area']} / {summary['deck_area']} "
            f"({summary['area_utilization']:.1%})",
            f"ShippingFee: {summary['shipping_fee']:g}",
            f"Objective: {summary['objective']:g}", "",
        ])
    outside = ", ".join(placement.vehicle.id for placement in state.outside_placements) or "-"
    blocks.extend([f"Outside ({len(state.outside_placements)}): {outside}", "",
                   "H: width x length asli", "V: width dan length ditukar", "Titik (0,0): kiri bawah"])
    lines = [line for block in blocks for line in (textwrap.wrap(block, 44) or [""])]
    fig = plt.figure(figsize=(12, max(5 * len(state.ships), len(lines) * .24)))
    layout = fig.add_gridspec(len(state.ships), 2, width_ratios=[1.5, 1])
    for index, (ship_id, ship) in enumerate(state.ships.items()):
        ax = fig.add_subplot(layout[index, 0])
        ax.set(xlim=(0, ship.width), ylim=(0, ship.length),
               xlabel="x / lebar (grid)", ylabel="y / panjang (grid)", title=ship_id)
        ax.set_aspect("equal")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=15, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=20, integer=True))
        ax.grid(alpha=.25)
        ax.set_axisbelow(True)
        for placement in state.inside_placements:
            if placement.ship_id != ship_id:
                continue
            ax.add_patch(Rectangle(
                (placement.x, placement.y), placement.placed_width, placement.placed_length,
                facecolor=colors[placement.vehicle.id], edgecolor="#253248", lw=1.2,
            ))
            label = f"{placement.vehicle.id}\n{'H' if placement.orientation.value == 'horizontal' else 'V'}"
            ax.text(placement.x + placement.placed_width / 2, placement.y + placement.placed_length / 2,
                    label, ha="center", va="center", fontsize=8, clip_on=True)
    info = fig.add_subplot(layout[:, 1])
    info.axis("off")
    info.text(0, 1, "\n".join(lines), va="top", fontsize=10, linespacing=1.5, transform=info.transAxes)
    fig.tight_layout()
    if save_path is not None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=150, bbox_inches="tight")
    return fig
