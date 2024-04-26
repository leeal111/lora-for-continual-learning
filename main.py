import argparse
import logging
import time
from os.path import exists
import numpy as np
from data_manager import DataManager
from model import load_vit_train_type
from test import compute_current_accuracy
from train import init_optimizer, init_routine, train
from utils import init_args, init_logging, weight_file_path

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
parser.add_argument("--tasks_lr_T", type=int, nargs="+")


parser.add_argument("--train_type", type=str, default="lora", choices=["lora"])

cfg = parser.parse_args()
init_args(cfg)
init_logging(cfg.log_path)

logging.info(f"====>")
logging.info(f"experiment settings:")
for arg_name, arg_value in cfg.__dict__.items():
    logging.info(f"{arg_name} : {str(arg_value)}")
logging.info(f"<====")

# load model
model = load_vit_train_type(cfg)

# load data_manager
data_manager = DataManager(cfg)

# routine start
init_routine(cfg)
model.to(cfg.device)

start_time = time.time()
known_class_num = 0
upper_accs = []

for task_index in range(cfg.tasks_num):

    # get class tag range
    current_class_num = cfg.class_num_per_task_list[task_index]
    accmulate_class_num = current_class_num + known_class_num
    logging.info(f" ")
    logging.info(
        f"====> Learn Task {task_index} with class range: {known_class_num}-{accmulate_class_num}"
    )

    # load data
    train_loader = data_manager.get_dataloader(
        cfg,
        np.arange(known_class_num, accmulate_class_num),
        source="train",
        mode="train",
    )
    test_loader = data_manager.get_dataloader(
        cfg, np.arange(0, accmulate_class_num), source="test", mode="test"
    )

    # 准备训练
    logging.info(f"====> Training")
    optimizer, scheduler = init_optimizer(cfg, model, task_index)
    fc_file_name, lora_file_name = weight_file_path(cfg, task_index)
    if exists(fc_file_name) and exists(lora_file_name):
        logging.info(f"load pth weight")
        model.load_lora_parameters(lora_file_name)
        model.to(cfg.device)
    else:
        logging.info(
            " || ".join(["epoch", "total_loss", "train_acc", "correct", "total", "lr"])
        )
        for epoch in range(1, cfg.epochs + 1):
            train(
                cfg,
                epoch,
                model,
                train_loader,
                optimizer,
                scheduler,
                known_class_num,
                accmulate_class_num,
            )
        model.save_fc_parameters(fc_file_name)
        model.save_lora_parameters(lora_file_name)
    logging.info(f"<==== Trained")

    # 最优上界分数
    logging.info(f"====> Testing")
    test_acc = compute_current_accuracy(
        cfg,
        model,
        test_loader,
        known_class_num,
        accmulate_class_num,
    )
    upper_accs.append(test_acc)

    logging.info(f"<==== Tested")
    known_class_num = accmulate_class_num

end_time = time.time()
logging.info(f"\n====> Total time: {(end_time - start_time)/60/60} h")
logging.info(f"\n====> Upper acc: {upper_accs}")
