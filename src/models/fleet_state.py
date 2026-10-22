from dataclasses import dataclass, field, replace
from typing import Any, Iterable

from .ship import Ship
from .state import Orientation, Placement, State
from .vehicle import Vehicle


@dataclass(slots=True)
class FleetState:
    ships: dict[str, Ship]
    placements: list[Placement] = field(default_factory=list)
    objective_value: float = 0.0
    total_weight: float = 0.0

    def __post_init__(self) -> None:
        self.ships = dict(self.ships)
        self.placements = list(self.placements)
        if not self.ships or any(
            not isinstance(ship_id, str) or not ship_id.strip() or not isinstance(ship, Ship)
            for ship_id, ship in self.ships.items()
        ):
            raise ValueError("Fleet requires ships with non-empty string ids")
        ids = [placement.vehicle.id for placement in self.placements]
        if len(ids) != len(set(ids)):
            raise ValueError("Vehicle ids must be unique within a fleet")
        if not self.has_valid_assignments():
            raise ValueError("Invalid fleet placement assignment")

    def has_valid_assignments(self) -> bool:
        for placement in self.placements:
            if type(placement.inside) is not bool or not isinstance(placement.orientation, Orientation):
                return False
            if placement.inside:
                if not isinstance(placement.ship_id, str) or placement.ship_id not in self.ships:
                    return False
                if type(placement.x) is not int or type(placement.y) is not int:
                    return False
            elif placement.ship_id is not None or placement.x is not None or placement.y is not None:
                return False
        return True

    @property
    def inside_placements(self) -> list[Placement]:
        return [placement for placement in self.placements if placement.inside]

    @property
    def outside_placements(self) -> list[Placement]:
        return [placement for placement in self.placements if not placement.inside]

    @property
    def vehicles(self) -> list[Vehicle]:
        return [placement.vehicle for placement in self.placements]

    def replace_placement(self, index: int, placement: Placement) -> "FleetState":
        placements = self.placements.copy()
        placements[index] = placement
        return self.replace_placements(placements)

    def replace_placements(self, placements: Iterable[Placement]) -> "FleetState":
        replacements = list(placements)
        if [placement.vehicle for placement in replacements] != self.vehicles:
            raise ValueError("Fleet replacement must preserve vehicles and their order")
        return replace(self, placements=replacements)

    def ship_state(self, ship_id: str) -> State:
        return State(
            self.ships[ship_id],
            [replace(placement, ship_id=None) for placement in self.inside_placements
             if placement.ship_id == ship_id],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_type": "multi_ship",
            "ships": [{"id": ship_id, **ship.to_dict()} for ship_id, ship in self.ships.items()],
            "placements": [
                {**placement.to_dict(), "shipId": placement.ship_id}
                for placement in self.placements
            ],
            "objectiveValue": self.objective_value,
            "totalWeight": self.total_weight,
        }


SearchState = State | FleetState
ShipConfiguration = Ship | dict[str, Ship]


def ship_configuration(state: SearchState) -> ShipConfiguration:
    return state.ships if isinstance(state, FleetState) else state.ship


def ship_items(state: SearchState) -> list[tuple[str | None, Ship]]:
    return list(state.ships.items()) if isinstance(state, FleetState) else [(None, state.ship)]


def state_key(state: SearchState) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (placement.vehicle.id, placement.inside, placement.ship_id,
         placement.x, placement.y, placement.orientation.value)
        for placement in state.placements
    )
