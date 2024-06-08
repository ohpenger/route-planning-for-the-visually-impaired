import sys
sys.path.append('..')
import xmltodict
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import List
from datetime import datetime
from request import request_post
import json

@dataclass_json
@dataclass
class Coordinate():
    lat: float
    lon: float
    time: int = 0


@dataclass_json
@dataclass
class Input_data():
    shape: List[Coordinate]
    costing: str = "pedestrian"
    shape_match: str = "walk_or_snap"


@dataclass
class Trace_attributes():
    file_path: str
    data: str = ""

    def load_data(self):
        with open(self.file_path, 'r') as f:
            xml_dict = xmltodict.parse(f.read())
        route_track = xml_dict["gpx"]["trk"]["trkseg"]["trkpt"]
        shape = []
        start_time = None
        previous_time = None
        for point in route_track:
            lat = float(point["@lat"])
            lon = float(point["@lon"])
            time = datetime.fromisoformat(point["time"][:-1])
            if start_time is None:
                start_time = time
            if previous_time is None:
                previous_time = time
            shape.append(Coordinate(lat, lon, time=(time - start_time).total_seconds()))

        shape[0].type = shape[-1].type = "break"
        data = Input_data(shape=shape)
        json_data = data.to_json()
        self.data = json_data
        return json_data

    def request1(self):
        url = "https://valhalla1.openstreetmap.de/trace_attributes"
        response = request_post(url, data=self.data)
        formatted_data = json.dumps(response, indent=2)
        return formatted_data
