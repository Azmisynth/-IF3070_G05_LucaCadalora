from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Ship:
    width: int
    length: int
    max_capacity: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.length <= 0:
            raise ValueError("Ship dimensions must be positive")
        if self.max_capacity < 0:
            raise ValueError("Ship capacity cannot be negative")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Ship":
        return cls(
            width=int(data["width"]),
            length=int(data["length"]),
            max_capacity=float(data.get("max_capacity", data.get("maxCapacity"))),
        )

    def to_dict(self) -> dict[str, int | float]:
        return {
            "width": self.width,
            "length": self.length,
            "maxCapacity": self.max_capacity,
        }
