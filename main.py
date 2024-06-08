import json
import tools
from collections import deque
import xmltodict
from myroutes import route
from myroutes import trace_route
from myroutes import trace_attributes

# method 1 to get route plan
# start_points = route.Coordinate(lat=49.59586959, lon=11.00270821)
# end_points = route.Coordinate(lat=49.59662267, lon=11.0070331)
# input_data = route.Input_data(locations=[start_points,end_points])
# route = route.Route(input_data)
# output = route.request()

# method 2 to get route plan
gps_file_path = "2023-12-04_4_Dec_2023_15_22_39.gpx"
trace_route = trace_route.Trace_route(file_path=gps_file_path)
trace_route.load_data()
output = trace_route.request()

# method 3 to get route plan
# trace_attributes = trace_attributes.Trace_attributes(file_path=gps_file_path)
# trace_attributes.load_data()
# output = trace_attributes.request()

start_location = {'lat': 49.59556625, 'lon':11.002758}
end_location = {'lat': 49.59591913, 'lon':11.0028471}

maneuvers = output['trip']['legs'][0]['maneuvers']
temp = json.dumps(maneuvers,indent=2)
total_length_valhalla = 0
valhalla_instruction_list = deque()
for maneuver in maneuvers:
    total_length_valhalla += maneuver["length"]
    valhalla_instruction_list.append({'length':maneuver['length'], 'instruction': maneuver['instruction']})




xml_result = open(gps_file_path, 'r')
xml_dict = xmltodict.parse(xml_result.read())
route_track = xml_dict["gpx"]["trk"]["trkseg"]["trkpt"]
previous_coordiante = {'lat': None, 'lon': None, 'current_time': None}
the_distance_you_have_walked = 0
valhalla_instruction = valhalla_instruction_list.popleft()
the_distance_you_got_to_turn = valhalla_instruction['length']
walk_simulator = tools.Walk_simulator(interval_time=3,file_path=gps_file_path)
for current_coordiante in walk_simulator.walking():
    distance_between_two_coordinate = tools.haversine(previous_coordiante['lat'],
                                              previous_coordiante['lon'],
                                              current_coordiante['lat'],
                                              current_coordiante['lon'])
    elapasd_time_between_two_coordinate = current_coordiante["elapasd_time_between_two_coordinate"]
    speed = 0 if elapasd_time_between_two_coordinate == 0 else distance_between_two_coordinate / elapasd_time_between_two_coordinate
    the_distance_you_have_walked += distance_between_two_coordinate
    if abs(the_distance_you_have_walked - the_distance_you_got_to_turn) < 0.01 :
        if valhalla_instruction_list:
            valhalla_instruction = valhalla_instruction_list.popleft()
            the_distance_you_got_to_turn += valhalla_instruction['length']
        else:
            break
        direction = tools.extract_directions(valhalla_instruction["instruction"])
        print(f'instruction: {direction}  lat: {current_coordiante["lat"]}    '
              f'lon: {current_coordiante["lon"]}  elapsed time: {current_coordiante["elapsed_time"]}'
              f' speed: {speed}')
    else:
        instruction = 'stay center'
        print(f'instruction: {instruction}  lat: {current_coordiante["lat"]}    '
              f'lon: {current_coordiante["lon"]}  elapsed time: {current_coordiante["elapsed_time"]}'
              f' speed: {speed}')
    previous_coordiante = current_coordiante

if valhalla_instruction_list:
    print(valhalla_instruction_list)