import os
import shutil


def write_intrinsics_line(file, id_, model, width, height, intrinsics):
    line = f"{id_} {model} {width} {height} {intrinsics[0][0]} {intrinsics[1][1]} {intrinsics[0][2]} {intrinsics[1][2]}\n"
    file.write(line)


def write_intrinsics_file_nuscenes(sample_path=None, sensor_params=None, img_width=1600, img_height=900,
                                   delete_temp=False, cameras=None):
    default_cameras = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

    cameras = default_cameras if cameras is None else cameras

    temp_file = os.path.join(os.getcwd(), "temp_cameras.txt")

    if delete_temp:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return

    colmap_intrinsics_file = os.path.join(sample_path, "cameras.txt")

    if os.path.exists(temp_file):
        shutil.copy(temp_file, colmap_intrinsics_file)
        return

    with open(temp_file, "w") as file:

        for idx, sensor in enumerate(cameras):
            idx += 1
            if sensor == 'CAM_FRONT':
                write_intrinsics_line(file, id_=idx, model='PINHOLE',
                                      width=img_width, height=img_height,
                                      intrinsics=sensor_params.CAM_FRONT['camera_intrinsic']
                                      )
            if sensor == 'CAM_FRONT_RIGHT':
                write_intrinsics_line(file, id_=idx, model='PINHOLE',
                                      width=img_width, height=img_height,
                                      intrinsics=sensor_params.CAM_FRONT_RIGHT['camera_intrinsic'])
            if sensor == 'CAM_BACK_RIGHT':
                write_intrinsics_line(file, id_=idx, model='PINHOLE',
                                      width=img_width, height=img_height,
                                      intrinsics=sensor_params.CAM_BACK_RIGHT['camera_intrinsic'])
            if sensor == 'CAM_BACK':
                write_intrinsics_line(file, id_=idx, model='PINHOLE',
                                      width=img_width, height=img_height,
                                      intrinsics=sensor_params.CAM_BACK['camera_intrinsic'])
            if sensor == 'CAM_BACK_LEFT':
                write_intrinsics_line(file, id_=idx, model='PINHOLE',
                                      width=img_width, height=img_height,
                                      intrinsics=sensor_params.CAM_BACK_LEFT['camera_intrinsic'])
            if sensor == 'CAM_FRONT_LEFT':
                write_intrinsics_line(file, id_=idx, model='PINHOLE',
                                      width=img_width, height=img_height,
                                      intrinsics=sensor_params.CAM_FRONT_LEFT['camera_intrinsic'])

    shutil.copy(temp_file, colmap_intrinsics_file)


def write_intrinsics_file_novelcam(intrinsics_folder_path, all_intrinsics, img_width=1600, img_height=900, ):
    intrinsics_file = os.path.join(intrinsics_folder_path, "intrinsics.txt")

    with open(intrinsics_file, "w") as file:
        for idx, intrinsics in enumerate(all_intrinsics):
            idx += 1
            write_intrinsics_line(file, id_=idx, model='PINHOLE',
                                  width=img_width, height=img_height,
                                  intrinsics=intrinsics)


def write_extrinsics_file_novelcam(extrinsics_folder_path, file_mode, transform_vectors):
    extrinsics_file = os.path.join(extrinsics_folder_path, "extrinsics.txt")

    img_idx = 1

    if file_mode == "a":
        with open(extrinsics_file, "r") as file:
            # Read all lines from the file
            lines = file.readlines()

            num_data_lines = len([line for line in lines if line.strip()[0] != '#'])

            # Extract the first character (assuming it's a number)
            img_idx = num_data_lines + 1

    with open(extrinsics_file, file_mode) as file:
        for idx, tv in enumerate(transform_vectors):
            line = f"{img_idx + idx} {tv[0]} {tv[1]} {tv[2]} {tv[3]} {tv[4]} {tv[5]} {tv[6]} {idx + 1}\n"
            file.write(line)


def win_path(_path):
    windows_path = _path.replace('/', '\\')
    if windows_path[1] == ":":
        windows_path = windows_path[0].upper() + windows_path[1:]

    return windows_path


def write_batch_file(sample_folder, colmap_manual_sparse_folder, colmap_sparse_folder):
    project_path = win_path(sample_folder)
    input_path = win_path(colmap_manual_sparse_folder)
    output_path = win_path(colmap_sparse_folder)
    win_drive = project_path[0].lower()

    batchfile_path = os.path.join(project_path, "batch.bat")
    batch_commands = [
        '@echo off',
        f'echo Starting COLMAP processing at {project_path} ...',
        f'cd /{win_drive} "{project_path}"',
        'REM Command 1: Feature Extraction',
        'echo Extracting features...',
        'call colmap feature_extractor --database_path ".\\database.db" --image_path ".\\images"',
        '',
        'REM Command 2: Matching',
        'echo Matching images...',
        'call colmap exhaustive_matcher --database_path ".\\database.db"',
        '',
        'REM Command 3: Triangulation',
        'echo Triangulating points...',
        f'call colmap point_triangulator --database_path ".\\database.db" --image_path ".\\images" --input_path "{input_path}" --output_path "{output_path}"',
        '',
        'REM Command 4: Convert to text file',
        'echo Converting sparse output to .txt ...',
        f'call colmap model_converter  --input_path "{output_path}" --output_path "{output_path}" --output_type TXT',
        '',
        'echo COLMAP processing completed.'
    ]

    with open(batchfile_path, 'w') as file:
        file.write('\n'.join(batch_commands))
