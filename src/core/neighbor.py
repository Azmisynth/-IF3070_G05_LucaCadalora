from collections.abc import Iterator
from itertools import chain
from dataclasses import replace
import random

from src.core.constraints import is_valid_state
from src.core.objective import with_updated_metrics
from src.models.state import Orientation, Placement
from src.models.ship import Ship
from src.models.fleet_state import FleetState, SearchState, ship_items, state_key


def _find_index(state: SearchState, vehicle_id: str) -> int:
    for index, placement in enumerate(state.placements):
        if placement.vehicle.id == vehicle_id:
            return index
    raise ValueError(f"Vehicle not found: {vehicle_id}")


def move_vehicle(
    state: SearchState,
    vehicle_id: str,
    x: int,
    y: int,
    orientation: Orientation | str | None = None,
    objective_mode: str = "shipping_fee",
    ship_id: str | None = None,
) -> SearchState | None:
    index = _find_index(state, vehicle_id)
    current = state.placements[index]
    if isinstance(state, FleetState):
        ship_id = ship_id if ship_id is not None else current.ship_id
        if ship_id not in state.ships:
            return None
    elif ship_id is not None:
        return None
    placement = Placement(
        vehicle=current.vehicle,
        inside=True,
        x=x,
        y=y,
        orientation=orientation or current.orientation,
        ship_id=ship_id,
    )
    candidate = state.replace_placement(index, placement)
    if not is_valid_state(candidate):
        return None
    return with_updated_metrics(candidate, objective_mode)


def rotate_vehicle(
    state: SearchState,
    vehicle_id: str,
    objective_mode: str = "shipping_fee",
) -> SearchState | None:
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
        ship_id=current.ship_id,
    )
    candidate = state.replace_placement(index, placement)
    if not is_valid_state(candidate):
        return None
    return with_updated_metrics(candidate, objective_mode)


def swap_vehicles(
    state: SearchState,
    first_vehicle_id: str,
    second_vehicle_id: str,
    objective_mode: str = "shipping_fee",
) -> SearchState | None:
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
        ship_id=second.ship_id,
    )
    placements[second_index] = Placement(
        vehicle=second.vehicle,
        inside=first.inside,
        x=first.x if first.inside else None,
        y=first.y if first.inside else None,
        orientation=first.orientation,
        ship_id=first.ship_id,
    )
    candidate = state.replace_placements(placements)
    if not is_valid_state(candidate):
        return None
    return with_updated_metrics(candidate, objective_mode)


def _move_candidates(state: SearchState, objective_mode: str) -> Iterator[SearchState]:
    for ship_id, ship in ship_items(state):
        yield from _ship_move_candidates(state, objective_mode, ship_id, ship)


def _ship_move_candidates(
    state: SearchState,
    objective_mode: str,
    ship_id: str | None,
    ship: Ship,
) -> Iterator[SearchState]:
    for placement in state.placements:
        orientations = [placement.orientation]
        if not placement.inside or placement.ship_id != ship_id:
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
            for y in range(ship.length - placed_length + 1):
                for x in range(ship.width - placed_width + 1):
                    if (
                        placement.inside
                        and placement.ship_id == ship_id
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
                        ship_id=ship_id,
                    )
                    if candidate is not None:
                        yield candidate


def _rotate_candidates(state: SearchState, objective_mode: str) -> Iterator[SearchState]:
    for placement in state.inside_placements:
        candidate = rotate_vehicle(state, placement.vehicle.id, objective_mode)
        if candidate is not None:
            yield candidate


def _swap_candidates(state: SearchState, objective_mode: str) -> Iterator[SearchState]:
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


def unload_vehicle(
    state: SearchState,
    vehicle_id: str,
    objective_mode: str = "shipping_fee",
) -> SearchState | None:
    index = _find_index(state, vehicle_id)
    current = state.placements[index]
    if not current.inside:
        return None
    return with_updated_metrics(
        state.replace_placement(index, Placement.outside(current.vehicle)), objective_mode,
    )


def transfer_and_load(
    state: FleetState,
    vehicle_id: str,
    outside_vehicle_id: str,
    ship_id: str,
    x: int,
    y: int,
    orientation: Orientation | str | None = None,
    loading_orientation: Orientation | str | None = None,
    objective_mode: str = "shipping_fee",
) -> FleetState | None:
    first_index = _find_index(state, vehicle_id)
    second_index = _find_index(state, outside_vehicle_id)
    first = state.placements[first_index]
    second = state.placements[second_index]
    if (not first.inside or second.inside or first.ship_id == ship_id
            or ship_id not in state.ships):
        return None
    placements = state.placements.copy()
    placements[first_index] = replace(
        first, ship_id=ship_id, x=x, y=y, orientation=orientation or first.orientation,
    )
    placements[second_index] = replace(
        second, inside=True, ship_id=first.ship_id, x=first.x, y=first.y,
        orientation=loading_orientation or first.orientation,
    )
    candidate = state.replace_placements(placements)
    return with_updated_metrics(candidate, objective_mode) if is_valid_state(candidate) else None


def _transfer_and_load_candidates(state: FleetState, objective_mode: str) -> Iterator[FleetState]:
    if not state.outside_placements:
        return
    for first in state.inside_placements:
        for second in state.outside_placements:
            for ship_id, ship in state.ships.items():
                if ship_id == first.ship_id:
                    continue
                for orientation in Orientation:
                    width = first.vehicle.width if orientation is Orientation.HORIZONTAL else first.vehicle.length
                    length = first.vehicle.length if orientation is Orientation.HORIZONTAL else first.vehicle.width
                    for loading_orientation in Orientation:
                        loading = replace(second, inside=True, ship_id=first.ship_id,
                                          x=first.x, y=first.y, orientation=loading_orientation)
                        source = state.ship_state(first.ship_id)
                        source = source.replace_placements([
                            replace(loading, ship_id=None) if p.vehicle.id == first.vehicle.id else p
                            for p in source.placements
                        ])
                        if not is_valid_state(source):
                            continue
                        for y in range(ship.length - length + 1):
                            for x in range(ship.width - width + 1):
                                candidate = transfer_and_load(
                                    state, first.vehicle.id, second.vehicle.id, ship_id, x, y,
                                    orientation, loading_orientation, objective_mode,
                                )
                                if candidate is not None:
                                    yield candidate


def _fleet_candidates(state: SearchState, objective_mode: str) -> Iterator[SearchState]:
    if isinstance(state, FleetState):
        for placement in state.inside_placements:
            yield unload_vehicle(state, placement.vehicle.id, objective_mode)
        yield from _transfer_and_load_candidates(state, objective_mode)


def generate_neighbors(
    state: SearchState,
    objective_mode: str = "shipping_fee",
) -> list[SearchState]:
    neighbors: list[SearchState] = []
    seen = {state_key(state)}
    candidates = chain(
        _move_candidates(state, objective_mode),
        _rotate_candidates(state, objective_mode),
        _swap_candidates(state, objective_mode),
        _fleet_candidates(state, objective_mode),
    )
    for candidate in candidates:
        key = state_key(candidate)
        if key not in seen:
            seen.add(key)
            neighbors.append(candidate)
    return neighbors


def random_move(state: SearchState, generator: random.Random, objective_mode: str) -> SearchState | None:
    if not state.placements:
        return None
    placement = generator.choice(state.placements)
    orientation = generator.choice(list(Orientation))
    ship_id, ship = generator.choice(ship_items(state)) if isinstance(state, FleetState) else (None, state.ship)
    width = placement.vehicle.width if orientation is Orientation.HORIZONTAL else placement.vehicle.length
    length = placement.vehicle.length if orientation is Orientation.HORIZONTAL else placement.vehicle.width
    if width > ship.width or length > ship.length:
        return None
    return move_vehicle(
        state, placement.vehicle.id, generator.randrange(ship.width - width + 1),
        generator.randrange(ship.length - length + 1), orientation, objective_mode, ship_id,
    )


def random_rotation(state: SearchState, generator: random.Random, objective_mode: str) -> SearchState | None:
    if not state.inside_placements:
        return None
    return rotate_vehicle(state, generator.choice(state.inside_placements).vehicle.id, objective_mode)


def random_swap(state: SearchState, generator: random.Random, objective_mode: str) -> SearchState | None:
    if len(state.placements) < 2:
        return None
    first, second = generator.sample(state.placements, 2)
    return swap_vehicles(state, first.vehicle.id, second.vehicle.id, objective_mode)


def _random_unload(state: SearchState, generator: random.Random, objective_mode: str) -> SearchState | None:
    if not state.inside_placements:
        return None
    return unload_vehicle(state, generator.choice(state.inside_placements).vehicle.id, objective_mode)


def _random_transfer_and_load(state: FleetState, generator: random.Random, objective_mode: str) -> FleetState | None:
    if not state.inside_placements or not state.outside_placements:
        return None
    first = generator.choice(state.inside_placements)
    second = generator.choice(state.outside_placements)
    destinations = [(ship_id, ship) for ship_id, ship in state.ships.items() if ship_id != first.ship_id]
    if not destinations:
        return None
    ship_id, ship = generator.choice(destinations)
    orientation = generator.choice(list(Orientation))
    width = first.vehicle.width if orientation is Orientation.HORIZONTAL else first.vehicle.length
    length = first.vehicle.length if orientation is Orientation.HORIZONTAL else first.vehicle.width
    if width > ship.width or length > ship.length:
        return None
    return transfer_and_load(
        state, first.vehicle.id, second.vehicle.id, ship_id,
        generator.randrange(ship.width - width + 1), generator.randrange(ship.length - length + 1),
        orientation, generator.choice(list(Orientation)), objective_mode,
    )


def random_neighbor(
    state: SearchState,
    generator: random.Random,
    objective_mode: str = "shipping_fee",
    attempts: int = 100,
) -> SearchState | None:
    operations = (random_move, random_rotation, random_swap)
    if isinstance(state, FleetState):
        operations += (_random_unload, _random_transfer_and_load)
    for _ in range(attempts):
        candidate = generator.choice(operations)(state, generator, objective_mode)
        if candidate is not None and state_key(candidate) != state_key(state):
            return candidate
    return None
