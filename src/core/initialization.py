import random
from collections.abc import Iterable

from src.core.constraints import is_valid_state
from src.core.objective import with_updated_metrics
from src.models.ship import Ship
from src.models.state import Orientation, Placement, State
from src.models.vehicle import Vehicle


def create_outside_state(
    ship: Ship,
    vehicles: Iterable[Vehicle],
    objective_mode: str = "shipping_fee",
) -> State:
    state = State(ship=ship, placements=[Placement.outside(vehicle) for vehicle in vehicles])
    return with_updated_metrics(state, objective_mode)


def generate_random_state(
    ship: Ship,
    vehicles: Iterable[Vehicle],
    objective_mode: str = "shipping_fee",
    max_attempts: int = 100,
    inclusion_probability: float = 0.8,
    rng: random.Random | None = None,
) -> State:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if not 0 <= inclusion_probability <= 1:
        raise ValueError("inclusion_probability must be between 0 and 1")

    generator = rng or random.Random()
    vehicle_list = list(vehicles)
    state = create_outside_state(ship, vehicle_list, objective_mode)
    indices = list(range(len(vehicle_list)))
    generator.shuffle(indices)

    for index in indices:
        vehicle = vehicle_list[index]
        if generator.random() > inclusion_probability:
            continue
        if state.total_weight + vehicle.weight > ship.max_capacity:
            continue

        orientations = [
            orientation
            for orientation in (Orientation.HORIZONTAL, Orientation.VERTICAL)
            if (
                vehicle.width if orientation is Orientation.HORIZONTAL else vehicle.length
            )
            <= ship.width
            and (
                vehicle.length if orientation is Orientation.HORIZONTAL else vehicle.width
            )
            <= ship.length
        ]
        if vehicle.width == vehicle.length and orientations:
            orientations = orientations[:1]
        if not orientations:
            continue

        for _ in range(max_attempts):
            orientation = generator.choice(orientations)
            placed_width = vehicle.width if orientation is Orientation.HORIZONTAL else vehicle.length
            placed_length = vehicle.length if orientation is Orientation.HORIZONTAL else vehicle.width
            placement = Placement(
                vehicle=vehicle,
                inside=True,
                x=generator.randrange(ship.width - placed_width + 1),
                y=generator.randrange(ship.length - placed_length + 1),
                orientation=orientation,
            )
            candidate = state.replace_placement(index, placement)
            if is_valid_state(candidate):
                state = with_updated_metrics(candidate, objective_mode)
                break

    return state
