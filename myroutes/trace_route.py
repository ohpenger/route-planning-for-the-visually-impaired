import xmltodict
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import List
import sys
sys.path.append("..")
from request import request_post


@dataclass_json
@dataclass
class Coordinate():
    lat: float
    lon: float
    type: str = "via"


@dataclass_json
@dataclass
class Input_data():
    shape: List[Coordinate]
    costing: str = "pedestrian"
    shape_match: str = "map_snap"


@dataclass
class Trace_route():
    file_path: str
    data: str = ""

    def load_data(self):
        with open(self.file_path, 'r') as f:
            xml_dict = xmltodict.parse(f.read())
        route_track = xml_dict["gpx"]["trk"]["trkseg"]["trkpt"]
        shape = []
        for point in route_track:
            lat = float(point["@lat"])
            lon = float(point["@lon"])
            shape.append(Coordinate(lat,lon,type= "via"))
        shape[0].type = shape[-1].type = "break"
        data = Input_data(shape = shape)
        json_data = data.to_json()
        self.data = json_data
        return json_data

    def request(self):
        url = "https://valhalla1.openstreetmap.de/trace_route"
        response = request_post(url,data=self.data)
        return response

