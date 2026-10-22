import json
import math
from pathlib import Path

from src.models.fleet_state import ShipConfiguration
from src.models.ship import Ship
from src.models.vehicle import Vehicle


def _validate_numbers(item: dict) -> None:
    if not isinstance(item, dict):
        raise ValueError("Kapal dan kendaraan harus berupa object")
    for key, value in item.items():
        if key == "id":
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} harus berupa angka finite")
    for key in ("width", "length"):
        if type(item.get(key)) is not int or item[key] <= 0:
            raise ValueError(f"{key} harus integer positif")


def parse_ships(items: list) -> dict[str, Ship]:
    if not isinstance(items, list) or not items:
        raise ValueError("ships harus berupa array kapal yang tidak kosong")
    ships = {}
    for item in items:
        _validate_numbers(item)
        ship_id = item.get("id")
        if not isinstance(ship_id, str) or not ship_id.strip() or ship_id in ships:
            raise ValueError("ID kapal harus berupa string unik dan tidak kosong")
        ships[ship_id] = Ship.from_dict(item)
    return ships


def load_problem(path: str | Path) -> tuple[ShipConfiguration, list[Vehicle]]:
    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or not isinstance(data.get("vehicles"), list):
        raise ValueError("Input memerlukan ship atau ships dan array vehicles")
    if ("ship" in data) == ("ships" in data):
        raise ValueError("Gunakan tepat satu dari ship atau ships")
    if "ships" in data:
        ships = parse_ships(data["ships"])
    else:
        _validate_numbers(data["ship"])
        ships = Ship.from_dict(data["ship"])
    for item in data["vehicles"]:
        _validate_numbers(item)
    vehicles = [Vehicle.from_dict(item) for item in data["vehicles"]]
    if not vehicles:
        raise ValueError("Minimal satu kendaraan diperlukan")
    if len({vehicle.id for vehicle in vehicles}) != len(vehicles):
        raise ValueError("ID kendaraan harus unik")
    return ships, vehicles
