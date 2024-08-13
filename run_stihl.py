import numpy as np
import torch
import argparse
import shutil
import os

from src import config
from src.slam import SLAM
from src.datasets import get_dataset


import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def backup_source_code(backup_directory):
    ignore_hidden = shutil.ignore_patterns(
        '.', '..', '.git*', '*pycache*', '*build', '*.fuse*', '*_drive_*',
        '*pretrained*', '*output*', '*media*', '*.so', '*.pyc', '*.Python',
        '*.eggs*', '*.DS_Store*', '*.idea*', '*.pth', '*__pycache__*', '*.ply',
        '*exps*',
    )

    if os.path.exists(backup_directory):
        shutil.rmtree(backup_directory)

    shutil.copytree('.', backup_directory, ignore=ignore_hidden)
    os.system('chmod -R g+w {}'.format(backup_directory))


SEQUENCES = {
    "kwald/drosselweg/flaeche1": [
        "2023-08-18", "2023-09-15", "2024-01-13", "2024-04-11", "2024-05-29_1", "2024-05-29_2", "2024-05-29_3", "2024-05-29_4"],
    "kwald/drosselweg/flaeche2": [
        "2023-08-18", "2023-12-21", "2024-01-13", "2024-04-11", "2024-05-29_1", "2024-05-30_1", "2024-05-30_2"],
    "esslingen/hse_dach": [
        "2023-07-20", "2023-11-07", "2024-01-27", "2024-04-14"],
    "esslingen/hse_hinterhof": [
        "2023-07-31", "2023-11-07", "2024-04-14", "2024-05-08", "2024-05-13_1", "2024-05-13_2", "2024-05-24_2"],
    "esslingen/hse_sporthalle": [
        "2023-09-11", "2023-11-23", "2024-02-19", "2024-04-14", "2024-05-07", "2024-05-08_1", "2024-05-08_2", "2024-05-24_1"],
}


if __name__ == '__main__':
    setup_seed(43)

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--max_frames", type=int, default=-1, help="Only [0, max_frames] Frames will be run")
    parser.add_argument("--only_tracking", action="store_true", help="Only tracking is triggered")
    parser.add_argument("--make_video", action="store_true", help="to generate video as in our project page")
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument("--image_size", nargs='+', default=None,
                        help='image height and width, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--calibration_txt', type=str, default=None,
                        help='calibration parameters: fx, fy, cx, cy, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--mode', type=str,
                        help='slam mode: mono, rgbd or stereo')
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    cfg = config.load_config(
        args.config, './configs/go_slam.yaml'
    )
    
    base_data_path = args.input_folder
    base_output_path = args.output
    
    if args.mode is not None:
        cfg['mode'] = args.mode
    if args.only_tracking:
        cfg['only_tracking'] = True
    if args.image_size is not None:
        cfg['cam']['H'], cfg['cam']['W'] = args.image_size
    if args.calibration_txt is not None:
        cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy'] = np.loadtxt(args.calibration_txt).tolist()

    assert cfg['mode'] in ['rgbd', 'mono', 'stereo'], cfg['mode']
    print(f"\n\n** Running {cfg['data']['input_folder']} in {cfg['mode']} mode!!! **\n\n")
    
    errors = dict()

    for location, dates in SEQUENCES.items():
        for date in dates:
            print(f"Running Location: {location} - Date: {date}")
            
            args.input_folder = os.path.join(base_data_path, location, date, "tum", "d435i")
            args.output = os.path.join(base_output_path, location, date, args.mode, "d435i")

            if args.output is None:
                output_dir = cfg['data']['output']
            else:
                output_dir = args.output

            # backup_source_code(os.path.join(output_dir, 'code'))
            # without backup_source_code function, the output_dir structure has to be created here:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir)

            config.save_config(cfg, f'{output_dir}/cfg.yaml')

            dataset = get_dataset(cfg, args, device=args.device)
            
            slam = SLAM(args, cfg)

            try:
                slam.run(dataset)
            except Exception as e:
                errors[f"{location} - {date}"] =  str(e)
                continue

            slam.terminate(rank=-1, stream=dataset)

            print('Done!')

