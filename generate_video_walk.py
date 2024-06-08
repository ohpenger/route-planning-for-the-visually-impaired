import json
import tools
from collections import deque
import xmltodict
from myroutes import route
from myroutes import trace_route
from myroutes import trace_attributes
import sys
from moviepy.editor import *
from moviepy.video.tools.subtitles import SubtitlesClip

gps_file_path = "2023-12-04_4_Dec_2023_15_22_39.gpx"
video_path = 'files/walking_video.mp4'
edited_video_path = "edited_walking_video.mp4"
script_name = sys.argv[0]
arguments = sys.argv[1:]
if len(arguments) < 2 or len(arguments) > 3:
    print('wrong arguments. Please check your arguments!')
    sys.exit(1)
elif len(arguments) == 2:
    video_path = arguments[0]
    edited_video_path = arguments[1]
elif len(arguments) == 3:
    video_path = arguments[0]
    edited_video_path = arguments[1]
    gps_file_path = arguments[2]


# method 2 to get route plan
trace_route = trace_route.Trace_route(file_path=gps_file_path)
trace_route.load_data()
output = trace_route.request()

maneuvers = output['trip']['legs'][0]['maneuvers']

total_length_valhalla = 0
estimated_time = 0
certain_distance = 0.01
valhalla_instruction_list = deque()
for maneuver in maneuvers:
    estimated_time += maneuver['time']
    total_length_valhalla += maneuver["length"]
    valhalla_instruction_list.append({'length':maneuver['length'], 'instruction': maneuver['instruction']})

xml_result = open(gps_file_path, 'r')
xml_dict = xmltodict.parse(xml_result.read())
route_track = xml_dict["gpx"]["trk"]["trkseg"]["trkpt"]
previous_coordinate = {'lat': None, 'lon': None, 'elapsed_time': 0.0,
                       'elapasd_time_between_two_coordinate': 0.0}
the_distance_you_have_walked = 0
speed = 0
valhalla_instruction = valhalla_instruction_list.popleft()
the_distance_you_got_to_turn = valhalla_instruction['length']
walk_simulator = tools.Walk_simulator(interval_time=3,file_path=gps_file_path)
command_list = []
for current_coordinate in walk_simulator.walking():
    distance_between_two_coordinate = tools.haversine(previous_coordinate['lat'],
                                              previous_coordinate['lon'],
                                              current_coordinate['lat'],
                                              current_coordinate['lon'])
    elapasd_time_between_two_coordinate = current_coordinate["elapasd_time_between_two_coordinate"]
    if previous_coordinate['lat'] != current_coordinate['lat']:
        speed = 0 if elapasd_time_between_two_coordinate == 0 \
            else round((distance_between_two_coordinate / elapasd_time_between_two_coordinate)*1000,2)
    the_distance_you_have_walked += distance_between_two_coordinate
    instruction = valhalla_instruction["instruction"]
    # it will be seen as a signal to turn around if the difference is less than a certain distance
    # certain distance here is 0.01
    if abs(the_distance_you_have_walked - the_distance_you_got_to_turn) < certain_distance:
        if valhalla_instruction_list:
            valhalla_instruction = valhalla_instruction_list.popleft()
            instruction = valhalla_instruction["instruction"]
            the_distance_you_got_to_turn += valhalla_instruction['length']
        else:
            break
        transformed_command = tools.extract_directions(instruction)
    else:
        transformed_command = 'stay center'
    if speed == 0:
        command = ' '
    else:
        command = f'instruction: {instruction}\n' \
            f'lat: {current_coordinate["lat"]}\n' \
            f'lon: {current_coordinate["lon"]}\n' \
            f'command: {transformed_command}\n' \
            f'estimated_time: {estimated_time}\n' \
            f'elapsed time: {current_coordinate["elapsed_time"]}s\n' \
            f'speed: {speed}m/s'
    command_list.append(((previous_coordinate['elapsed_time'],current_coordinate['elapsed_time']),command))
    previous_coordinate = current_coordinate

generator = lambda txt: TextClip(txt, font='Arial', fontsize=16, color='black')

subtitles = SubtitlesClip(command_list, generator)

video = VideoFileClip(video_path)
result = CompositeVideoClip([video, subtitles.set_pos(('left','top'))])

result.write_videofile(edited_video_path, fps=video.fps, temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")


