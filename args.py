from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument('--root_data_dir', type=str, default=f"{os.getcwd()}/data")
parser.add_argument('--exp_name', type=str, default='debug')
parser.add_argument('--dataset_name', type=str, default='hit-uav')
parser.add_argument('--add_background_label', action='store_true', default=True)

args = parser.parse_args()