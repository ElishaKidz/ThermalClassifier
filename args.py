from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument('--root_data_dir', type=str, default="/home/xd_eshaar_gcp_idf_il/data")
parser.add_argument('--exp_name', type=str, default='refactor')
parser.add_argument('--train_datasets_names', type=str, default='hit-uav')
parser.add_argument('--val_datasets_names', type=str, default='hit-uav')
parser.add_argument('--test_datasets_names', type=str, default='hit-uav')
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--devices', type=int, default=1)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--add_background_label', action='store_true', default=True)

args = parser.parse_args()