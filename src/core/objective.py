from dataclasses import replace
from typing import Callable

from src.models.state import Placement, State


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


def calculate_objective(state: State, mode: str = "shipping_fee") -> float:
    try:
        score = OBJECTIVE_MODES[mode]
    except KeyError as error:
        choices = ", ".join(OBJECTIVE_MODES)
        raise ValueError(f"Unknown objective mode: {mode}. Choose from {choices}") from error
    return sum(score(placement) for placement in state.inside_placements)


def calculate_total_weight(state: State) -> float:
    return sum(placement.vehicle.weight for placement in state.inside_placements)


def with_updated_metrics(state: State, mode: str = "shipping_fee") -> State:
    return replace(
        state,
        objective_value=calculate_objective(state, mode),
        total_weight=calculate_total_weight(state),
    )
