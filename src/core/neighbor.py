from collections.abc import Iterator
from itertools import chain

from src.core.constraints import is_valid_state
from src.core.objective import with_updated_metrics
from src.models.state import Orientation, Placement, State


def _find_index(state: State, vehicle_id: str) -> int:
    for index, placement in enumerate(state.placements):
        if placement.vehicle.id == vehicle_id:
            return index
    raise ValueError(f"Vehicle not found: {vehicle_id}")


def move_vehicle(
    state: State,
    vehicle_id: str,
    x: int,
    y: int,
    orientation: Orientation | str | None = None,
    objective_mode: str = "shipping_fee",
) -> State | None:
    index = _find_index(state, vehicle_id)
    current = state.placements[index]
    placement = Placement(
        vehicle=current.vehicle,
        inside=True,
        x=x,
        y=y,
        orientation=orientation or current.orientation,
    )
    candidate = state.replace_placement(index, placement)
    if not is_valid_state(candidate):
        return None
    return with_updated_metrics(candidate, objective_mode)


def rotate_vehicle(
    state: State,
    vehicle_id: str,
    objective_mode: str = "shipping_fee",
) -> State | None:
    index = _find_index(state, vehicle_id)
    current = state.placements[index]
    if not current.inside or current.x is None or current.y is None:
        return None
    placement = Placement(
        vehicle=current.vehicle,
        inside=True,
        x=current.x,
        y=current.y,
        orientation=current.orientation.rotated(),
    )
    candidate = state.replace_placement(index, placement)
    if not is_valid_state(candidate):
        return None
    return with_updated_metrics(candidate, objective_mode)


def swap_vehicles(
    state: State,
    first_vehicle_id: str,
    second_vehicle_id: str,
    objective_mode: str = "shipping_fee",
) -> State | None:
    first_index = _find_index(state, first_vehicle_id)
    second_index = _find_index(state, second_vehicle_id)
    if first_index == second_index:
        return None

    first = state.placements[first_index]
    second = state.placements[second_index]
    if not first.inside and not second.inside:
        return None

    placements = state.placements.copy()
    placements[first_index] = Placement(
        vehicle=first.vehicle,
        inside=second.inside,
        x=second.x if second.inside else None,
        y=second.y if second.inside else None,
        orientation=second.orientation,
    )
    placements[second_index] = Placement(
        vehicle=second.vehicle,
        inside=first.inside,
        x=first.x if first.inside else None,
        y=first.y if first.inside else None,
        orientation=first.orientation,
    )
    candidate = state.replace_placements(placements)
    if not is_valid_state(candidate):
        return None
    return with_updated_metrics(candidate, objective_mode)


def _move_candidates(state: State, objective_mode: str) -> Iterator[State]:
    for placement in state.placements:
        orientations = [placement.orientation]
        if not placement.inside:
            orientations = [Orientation.HORIZONTAL, Orientation.VERTICAL]
        for orientation in orientations:
            placed_width = (
                placement.vehicle.width
                if orientation is Orientation.HORIZONTAL
                else placement.vehicle.length
            )
            placed_length = (
                placement.vehicle.length
                if orientation is Orientation.HORIZONTAL
                else placement.vehicle.width
            )
            for y in range(state.ship.length - placed_length + 1):
                for x in range(state.ship.width - placed_width + 1):
                    if (
                        placement.inside
                        and x == placement.x
                        and y == placement.y
                        and orientation is placement.orientation
                    ):
                        continue
                    candidate = move_vehicle(
                        state,
                        placement.vehicle.id,
                        x,
                        y,
                        orientation,
                        objective_mode,
                    )
                    if candidate is not None:
                        yield candidate


def _rotate_candidates(state: State, objective_mode: str) -> Iterator[State]:
    for placement in state.inside_placements:
        candidate = rotate_vehicle(state, placement.vehicle.id, objective_mode)
        if candidate is not None:
            yield candidate


def _swap_candidates(state: State, objective_mode: str) -> Iterator[State]:
    for first_index in range(len(state.placements)):
        for second_index in range(first_index + 1, len(state.placements)):
            candidate = swap_vehicles(
                state,
                state.placements[first_index].vehicle.id,
                state.placements[second_index].vehicle.id,
                objective_mode,
            )
            if candidate is not None:
                yield candidate


def _state_key(state: State) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            placement.vehicle.id,
            placement.inside,
            placement.x,
            placement.y,
            placement.orientation.value,
        )
        for placement in state.placements
    )


def generate_neighbors(
    state: State,
    objective_mode: str = "shipping_fee",
) -> list[State]:
    neighbors: list[State] = []
    seen: set[tuple[tuple[object, ...], ...]] = set()
    candidates = chain(
        _move_candidates(state, objective_mode),
        _rotate_candidates(state, objective_mode),
        _swap_candidates(state, objective_mode),
    )
    for candidate in candidates:
        key = _state_key(candidate)
        if key not in seen:
            seen.add(key)
            neighbors.append(candidate)
    return neighbors
