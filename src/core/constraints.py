from itertools import combinations

from src.models.state import Placement, State


def is_inside_boundary(state: State) -> bool:
    for placement in state.inside_placements:
        if placement.x is None or placement.y is None:
            return False
        if placement.x < 0 or placement.y < 0:
            return False
        if placement.x + placement.placed_width > state.ship.width:
            return False
        if placement.y + placement.placed_length > state.ship.length:
            return False
    return True


def placements_overlap(first: Placement, second: Placement) -> bool:
    if not first.inside or not second.inside:
        return False
    if first.x is None or first.y is None or second.x is None or second.y is None:
        return False
    return (
        first.x < second.x + second.placed_width
        and first.x + first.placed_width > second.x
        and first.y < second.y + second.placed_length
        and first.y + first.placed_length > second.y
    )


def has_overlap(state: State) -> bool:
    return any(
        placements_overlap(first, second)
        for first, second in combinations(state.inside_placements, 2)
    )


def within_capacity(state: State) -> bool:
    weight = sum(placement.vehicle.weight for placement in state.inside_placements)
    return weight <= state.ship.max_capacity


def is_valid_state(state: State) -> bool:
    return is_inside_boundary(state) and not has_overlap(state) and within_capacity(state)
