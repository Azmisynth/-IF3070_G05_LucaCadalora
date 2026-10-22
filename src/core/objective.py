from dataclasses import replace
from typing import Callable

from src.models.state import Placement
from src.models.fleet_state import FleetState, SearchState


def _shipping_fee(placement: Placement) -> float:
    return placement.vehicle.shipping_fee


def _fee_per_area(placement: Placement) -> float:
    return placement.vehicle.shipping_fee / placement.vehicle.area


def _fee_with_eta(placement: Placement) -> float:
    return placement.vehicle.shipping_fee / (1.0 + placement.vehicle.eta)


OBJECTIVE_MODES: dict[str, Callable[[Placement], float]] = {
    "shipping_fee": _shipping_fee,
    "fee_per_area": _fee_per_area,
    "fee_with_eta": _fee_with_eta,
}


def calculate_objective(state: SearchState, mode: str = "shipping_fee") -> float:
    try:
        score = OBJECTIVE_MODES[mode]
    except KeyError as error:
        choices = ", ".join(OBJECTIVE_MODES)
        raise ValueError(f"Unknown objective mode: {mode}. Choose from {choices}") from error
    return sum(score(placement) for placement in state.inside_placements)


def calculate_total_weight(state: SearchState) -> float:
    return sum(placement.vehicle.weight for placement in state.inside_placements)


def with_updated_metrics(state: SearchState, mode: str = "shipping_fee") -> SearchState:
    return replace(
        state,
        objective_value=calculate_objective(state, mode),
        total_weight=calculate_total_weight(state),
    )


def ship_summaries(state: FleetState, mode: str = "shipping_fee") -> list[dict]:
    from src.core.constraints import is_valid_state

    summaries = []
    for ship_id, ship in state.ships.items():
        local = state.ship_state(ship_id)
        area = sum(placement.vehicle.area for placement in local.inside_placements)
        weight = calculate_total_weight(local)
        summaries.append({
            "ship_id": ship_id,
            "loaded_ids": [placement.vehicle.id for placement in local.inside_placements],
            "loaded_count": len(local.inside_placements),
            "weight": weight,
            "capacity": ship.max_capacity,
            "weight_utilization": weight / ship.max_capacity if ship.max_capacity else 0.0,
            "occupied_area": area,
            "deck_area": ship.width * ship.length,
            "area_utilization": area / (ship.width * ship.length),
            "shipping_fee": calculate_objective(local),
            "objective": calculate_objective(local, mode),
            "valid": is_valid_state(local),
        })
    return summaries
