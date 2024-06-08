import tools
from collections import deque
import xmltodict
from myroutes import trace_route
from moviepy.editor import *
from moviepy.video.tools.subtitles import SubtitlesClip
import cv2
import argparse
from sensation.helper.segmentation import Segmentator
from sensation.utils.analyze import find_dominant_column


def process_video(input_path, gps_path, output_path, maneuvers, segmentator):
    cap = cv2.VideoCapture(input_path)
    total_length_valhalla = 0
    estimated_time = 0
    certain_distance = 0.01
    valhalla_instruction_list = deque()
    for maneuver in maneuvers:
        estimated_time += maneuver["time"]
        total_length_valhalla += maneuver["length"]
        if "street_names" in maneuver:
            valhalla_instruction_list.append(
                {
                    "length": maneuver["length"],
                    "instruction": maneuver["instruction"],
                    "street": maneuver["street_names"],
                }
            )
        else:
            valhalla_instruction_list.append(
                {
                    "length": maneuver["length"],
                    "instruction": maneuver["instruction"],
                    "street": None,
                }
            )

    xml_result = open(gps_path, "r")
    xml_dict = xmltodict.parse(xml_result.read())
    route_track = xml_dict["gpx"]["trk"]["trkseg"]["trkpt"]
    previous_coordinate = {
        "lat": None,
        "lon": None,
        "elapsed_time": 0.0,
        "elapasd_time_between_two_coordinate": 0.0,
    }
    the_distance_you_have_walked = 0
    speed = 0
    valhalla_instruction = valhalla_instruction_list.popleft()
    the_distance_you_got_to_turn = valhalla_instruction["length"]
    walk_simulator = tools.Walk_simulator(interval_time=3, file_path=gps_path)
    command_list = []

    for current_coordinate in walk_simulator.walking():
        distance_between_two_coordinate = tools.haversine(
            previous_coordinate["lat"],
            previous_coordinate["lon"],
            current_coordinate["lat"],
            current_coordinate["lon"],
        )
        elapasd_time_between_two_coordinate = current_coordinate[
            "elapasd_time_between_two_coordinate"
        ]
        if previous_coordinate["lat"] != current_coordinate["lat"]:
            speed = (
                0
                if elapasd_time_between_two_coordinate == 0
                else round(
                    (
                        distance_between_two_coordinate
                        / elapasd_time_between_two_coordinate
                    )
                    * 1000,
                    2,
                )
            )
        the_distance_you_have_walked += distance_between_two_coordinate
        instruction = valhalla_instruction["instruction"]
        current_street = valhalla_instruction["street"]

        cap.set(cv2.CAP_PROP_POS_MSEC, current_coordinate["elapsed_time"])
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = segmentator.inference(frame)
        maskrgb = segmentator.mask_to_rgb(mask)

        dominant_column = find_dominant_column(maskrgb, [(244, 35, 232)])
        # print(f"The dominant column is: {dominant_column}")
        # it will be seen as a signal to turn around if the difference is less than a certain distance
        # certain distance here is 0.01
        if (
            abs(the_distance_you_have_walked - the_distance_you_got_to_turn)
            < certain_distance
        ):
            if valhalla_instruction_list:
                valhalla_instruction = valhalla_instruction_list.popleft()
                instruction = valhalla_instruction["instruction"]
                current_street = valhalla_instruction["street"]
                the_distance_you_got_to_turn += valhalla_instruction["length"]
            else:
                break
            transformed_command = tools.extract_directions(instruction)
        else:
            if dominant_column == 1:
                transformed_command = "Go left"
            elif dominant_column == 2:
                transformed_command = "Stay center"
            elif dominant_column == 3:
                transformed_command = "Go right"
            else:
                print("No sidewalk detected in the frame.")
        if speed == 0:
            command = " "
        else:
            command = (
                f'instruction: {instruction}\n'
                f'lat: {current_coordinate["lat"]}\n'
                f'lon: {current_coordinate["lon"]}\n'
                f'command: {transformed_command}\n'
                f'estimated_time: {estimated_time}\n'
                f'elapsed time: {current_coordinate["elapsed_time"]}s\n'
                f'speed: {speed}m/s\n'
                f'current street: {current_street}'
            )

        command_list.append(
            (
                (
                    previous_coordinate["elapsed_time"],
                    current_coordinate["elapsed_time"],
                ),
                command,
            )
        )
        previous_coordinate = current_coordinate

    cv2.destroyAllWindows()
    cap.release()

    generator = lambda txt: TextClip(txt, font="Times-Bold", fontsize=18, color="black")

    subtitles = SubtitlesClip(command_list, generator)

    video = VideoFileClip(input_path)
    result = CompositeVideoClip([video, subtitles.set_pos(("left", "top"))])

    result.write_videofile(
        output_path,
        fps=video.fps,
        temp_audiofile="temp-audio.m4a",
        remove_temp=True,
        codec="libx264",
        audio_codec="aac",
    )


def main():
    parser = argparse.ArgumentParser(description="RSU_VI input handler")
    parser.add_argument("input_path", help="path to input video")
    parser.add_argument("gps_path", help="path to gps coordinates")
    parser.add_argument("output_path", help="path to output video")

    args = parser.parse_args()
    model_path = "./model_weights/deeplabv3resnet50.onnx"

    trace_route1 = trace_route.Trace_route(file_path=args.gps_path)
    trace_route1.load_data()
    output = trace_route1.request()
    maneuvers = output["trip"]["legs"][0]["maneuvers"]
    segmentator = Segmentator(model_path)

    process_video(
        args.input_path, args.gps_path, args.output_path, maneuvers, segmentator
    )


if __name__ == "__main__":
    main()
