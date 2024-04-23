import os
import platform

def output_trajectory_to_file(file_path_string, filename_string, trajectory_list):
    if not os.path.exists(file_path_string):
        os.makedirs(file_path_string)
    with open(file_path_string + '/' + filename_string, 'w') as file:
        for point in trajectory_list:
            if isinstance(point, int):
                file.write(str(point) + '\n')
            else:
                file.write(str(point.tolist()) + '\n')
