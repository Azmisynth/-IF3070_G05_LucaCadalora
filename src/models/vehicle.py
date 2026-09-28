from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Vehicle:
    id: str
    width: int
    length: int
    shipping_fee: float
    weight: float
    eta: float

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Vehicle id cannot be empty")
        if self.width <= 0 or self.length <= 0:
            raise ValueError("Vehicle dimensions must be positive")
        if self.shipping_fee < 0 or self.weight < 0 or self.eta < 0:
            raise ValueError("Vehicle values cannot be negative")

    @property
    def area(self) -> int:
        return self.width * self.length

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Vehicle":
        return cls(
            id=str(data["id"]),
            width=int(data["width"]),
            length=int(data["length"]),
            shipping_fee=float(data.get("shipping_fee", data.get("shippingFee"))),
            weight=float(data["weight"]),
            eta=float(data.get("eta", data.get("ETA", 0))),
        )

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "id": self.id,
            "width": self.width,
            "length": self.length,
            "shippingFee": self.shipping_fee,
            "weight": self.weight,
            "ETA": self.eta,
        }
