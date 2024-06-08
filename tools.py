import xmltodict
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
import re
import time
from queue import Queue



def haversine(lat1, lon1, lat2, lon2):

    if lat1 is None or lon1 is None:
        lat1 = lat2
        lon1 = lon2

    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Radius of the Earth in kilometers (you can change it to miles if needed)
    R = 6371.0

    # Calculate the distance
    distance = R * c

    return distance


def extract_directions(text):
    pattern = re.compile(r'(right|left|You have arrived at your destination)', re.IGNORECASE)
    matches = pattern.findall(text)
    if not matches:
        return "stay center"
    elif matches[0].startswith("You"):
        return matches[0]
    else:
        return "go " + matches[0]


class Walk_simulator():
    interval_time: int
    file_path: str
    start_time: datetime
    route_track: Queue

    def __init__(self,file_path,interval_time):
        self.file_path = file_path
        self.interval_time = interval_time
        self.route_track = Queue()
        with open(self.file_path, 'r') as f:
            xml_dict = xmltodict.parse(f.read())
        temp_data = xml_dict["gpx"]["trk"]["trkseg"]["trkpt"]
        for temp in temp_data:
            self.route_track.put(temp)

    def walking(self):
        location = self.route_track.get()
        self.start_time = datetime.fromisoformat(location["time"][:-1])
        previous_time = self.start_time
        current_time = self.start_time
        elapsed_time = (current_time - self.start_time).total_seconds()
        gps_point_time = datetime.fromisoformat(location["time"][:-1])
        gps_elapsed_time = gps_point_time - self.start_time
        while not self.route_track.empty():
            if elapsed_time + self.interval_time >= gps_elapsed_time.total_seconds():
                # time.sleep(gps_elapsed_time.total_seconds() - elapsed_time)
                lat = float(location["@lat"])
                lon = float(location["@lon"])
                elapsed_time = gps_elapsed_time.total_seconds()
                elapasd_time_between_two_coordinate = (gps_point_time - previous_time).total_seconds()
                yield {'lat': lat, 'lon': lon, 'elapsed_time': elapsed_time, 'elapasd_time_between_two_coordinate': elapasd_time_between_two_coordinate}
                location = self.route_track.get()
                previous_time = gps_point_time
                gps_point_time = datetime.fromisoformat(location["time"][:-1])
                gps_elapsed_time = gps_point_time - self.start_time
            else:
                # time.sleep(self.interval_time)
                lat = float(location["@lat"])
                lon = float(location["@lon"])
                elapsed_time += self.interval_time
                elapasd_time_between_two_coordinate = (gps_point_time - previous_time).total_seconds()
                yield {'lat': lat, 'lon': lon, 'elapsed_time': elapsed_time, 'elapasd_time_between_two_coordinate': elapasd_time_between_two_coordinate}



