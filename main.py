import argparse
import logging
from model import load_vit_train_type
from utils import init_args

# [120, 48, 125, 24, 6]
# ["gaugan", "biggan", "wild", "whichfaceisreal", "san"]

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--workers_num", type=int, default=8)
parser.add_argument("--gpus", type=int, nargs="+", default=[1])
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--rank", type=int, default=10)

parser.add_argument("--n_clusters", type=int, default=5)
parser.add_argument("--dataset_path", type=str, default="./datas/CDDB")
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--pretrain_model_path", type=str, default="./pths/B_16.pth")
parser.add_argument("--pretrain_model_name", type=str)
parser.add_argument("--classes_num", type=int, default=10)
parser.add_argument("--tasks_num", type=int, default=5)
parser.add_argument("--result_path", type=str, default="./results")

parser.add_argument("--class_num_per_task_list", nargs="+")
parser.add_argument("--tasks_name", nargs="+")
parser.add_argument("--tasks_lr_T", nargs="+")


parser.add_argument("--train_type", type=str, default="lora", choices=["lora"])

cfg = parser.parse_args()
init_args(cfg)

logging.info(f"====>")
logging.info(f"experiment settings:")
for arg_name, arg_value in cfg.__dict__.items():
    logging.info(f"{arg_name} : {str(arg_value)}")
logging.info(f"<====")

# load model 
model = load_vit_train_type(cfg)
