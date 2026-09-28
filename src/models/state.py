from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable

from .ship import Ship
from .vehicle import Vehicle


class Orientation(str, Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"

    def rotated(self) -> "Orientation":
        if self is Orientation.HORIZONTAL:
            return Orientation.VERTICAL
        return Orientation.HORIZONTAL


@dataclass(frozen=True, slots=True)
class Placement:
    vehicle: Vehicle
    inside: bool = False
    x: int | None = None
    y: int | None = None
    orientation: Orientation = Orientation.HORIZONTAL

    def __post_init__(self) -> None:
        if isinstance(self.orientation, str):
            object.__setattr__(self, "orientation", Orientation(self.orientation.lower()))
        if self.inside and (self.x is None or self.y is None):
            raise ValueError("Inside placement requires coordinates")

    @property
    def placed_width(self) -> int:
        if self.orientation is Orientation.HORIZONTAL:
            return self.vehicle.width
        return self.vehicle.length

    @property
    def placed_length(self) -> int:
        if self.orientation is Orientation.HORIZONTAL:
            return self.vehicle.length
        return self.vehicle.width

    @classmethod
    def outside(cls, vehicle: Vehicle) -> "Placement":
        return cls(vehicle=vehicle)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vehicle": self.vehicle.to_dict(),
            "inside": self.inside,
            "x": self.x,
            "y": self.y,
            "orientation": self.orientation.value,
        }


@dataclass(slots=True)
class State:
    ship: Ship
    placements: list[Placement] = field(default_factory=list)
    objective_value: float = 0.0
    total_weight: float = 0.0

    def __post_init__(self) -> None:
        self.placements = list(self.placements)
        ids = [placement.vehicle.id for placement in self.placements]
        if len(ids) != len(set(ids)):
            raise ValueError("Vehicle ids must be unique within a state")

    @property
    def inside_placements(self) -> list[Placement]:
        return [placement for placement in self.placements if placement.inside]

    @property
    def outside_placements(self) -> list[Placement]:
        return [placement for placement in self.placements if not placement.inside]

    @property
    def vehicles(self) -> list[Vehicle]:
        return [placement.vehicle for placement in self.placements]

    def replace_placement(self, index: int, placement: Placement) -> "State":
        placements = self.placements.copy()
        placements[index] = placement
        return replace(self, placements=placements)

    def replace_placements(self, placements: Iterable[Placement]) -> "State":
        return replace(self, placements=list(placements))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ship": self.ship.to_dict(),
            "placements": [placement.to_dict() for placement in self.placements],
            "objectiveValue": self.objective_value,
            "totalWeight": self.total_weight,
        }
