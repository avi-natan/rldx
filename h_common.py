import json
import os

def output_trajectory_to_file(file_path_string, filename_string, trajectory_list):
    if not os.path.exists(file_path_string):
        os.makedirs(file_path_string)
    with open(file_path_string + '/' + filename_string, 'w') as file:
        for point in trajectory_list:
            if isinstance(point, int):
                file.write(str(point) + '\n')
            else:
                file.write(str(point.tolist()) + '\n')

def read_json_data(params_file):
    with open(params_file, 'r') as file:
        json_data = json.load(file)
    return json_data
