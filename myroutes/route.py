from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import List
from typing import Dict
import sys
sys.path.append("..")
from request import request_post

@dataclass_json
@dataclass
class Coordinate():
    lat: float
    lon: float

@dataclass_json
@dataclass
class Input_data():
    locations: List[Coordinate]
    directions_options: Dict[str, str]
    costing: str = "pedestrian"

    def __init__(self,locations,directions_option = {"units": "miles"},costing = "pedestrian"):
        self.locations = locations
        self.directions_options = directions_option
        self.costing = costing


@dataclass
class Route():
    input_data : Input_data

    def request(self):
        url = "https://valhalla1.openstreetmap.de/route"
        json_data = self.input_data.to_json()
        response = request_post(url,json_data)
        return response