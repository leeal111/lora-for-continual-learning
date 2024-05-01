import argparse
import logging
import time
from os.path import exists
import numpy as np
from data_manager import DataManager
from lora import set_task_index
from model import load_vit_train_type
from test import compute_current_accuracy, eval_cnn
from train import clustering, init_optimizer, init_other, train
from utils import argv_str, center_file_path, init_args, init_logging, weight_file_path
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--test_label", action="store_true")
parser.add_argument("--batch_size", type=int, default=50)
parser.add_argument("--workers_num", type=int, default=2)
parser.add_argument("--gpus", type=int, nargs="+", default=[3])
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--lr", type=float, default=3e-3)
parser.add_argument("--epochs", type=int, default=13)
parser.add_argument("--rank", type=int, default=10)

parser.add_argument("--n_clusters", type=int, default=5)
parser.add_argument("--dataset_path", type=str, default="./datas/CDDB")
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--pretrain_model_path", type=str, default="./pths/B_16.pth")
parser.add_argument("--pretrain_model_name", type=str)
parser.add_argument("--classes_num", type=int, default=10)
parser.add_argument("--tasks_num", type=int, default=5)
parser.add_argument("--result_path", type=str, default="./results")

parser.add_argument("--class_num_per_task_list", type=int, nargs="+")
parser.add_argument(
    "--tasks_name",
    nargs="+",
    default=["gaugan", "biggan", "wild", "whichfaceisreal", "san"],
)
parser.add_argument("--tasks_lr_T", type=int, nargs="+", default=[120, 48, 125, 24, 6])


parser.add_argument("--train_type", type=str, default="lora", choices=["lora"])
parser.add_argument("--enable_load_weight", action="store_true")
parser.add_argument("--enable_load_center", action="store_true")
parser.add_argument("--not_upper_test", action="store_true")
parser.add_argument("--not_cluster", action="store_true")
parser.add_argument("--not_eval", action="store_true")
parser.add_argument("--raitolossA", type=float, default=0)
parser.add_argument("--raitolossB", type=float, default=0)
parser.add_argument("--loss_ratio_start", type=float, default=1)
parser.add_argument("--loss_ratio_end", type=float, default=1)
parser.add_argument("--ratio_type", type=str, default="linear")
parser.add_argument("--raitolossTeacher", type=float, default=0)

args = parser.parse_args()

init_args(args)

init_logging(args.log_path, file_name=argv_str(sys.argv[1:]))
init_other(args)

logging.info(f"experiment settings:")
for arg_name, arg_value in args.__dict__.items():
    logging.info(f"{arg_name} : {str(arg_value)}")
logging.info(f"  ")

# load model
model = load_vit_train_type(args)
model.to(args.device)

# load data_manager
data_manager = DataManager(args)


start_time = time.time()
known_class_num = 0
upper_accs = []
kmeans_centers = []
tasks_accs = []
mean_accs = []
total_nums = []
for task_index in range(args.tasks_num):
    task_start_time = time.time()

    # get class tag range
    current_class_num = args.class_num_per_task_list[task_index]
    accmulate_class_num = current_class_num + known_class_num
    logging.info(f"  ")
    logging.info(f"====>   ")
    logging.info(
        f"====> {task_index} Task Learn with class range: {known_class_num}-{accmulate_class_num}"
    )
    logging.info(f"====>   ")
    logging.info(f"  ")

    # load data
    train_loader = data_manager.get_dataloader(
        args,
        np.arange(known_class_num, accmulate_class_num),
        source="train",
        mode="train",
    )
    test_loader = data_manager.get_dataloader(
        args, np.arange(0, accmulate_class_num), source="test", mode="test"
    )

    # 准备训练
    optimizer, scheduler = init_optimizer(args, model, task_index)
    logging.info(f"  ")
    logging.info(f"====> {task_index} Training")
    lora_file_name = weight_file_path(args, task_index)
    if args.enable_load_weight and exists(lora_file_name):
        logging.info(f"load pth weight")
        model.load_lora_parameters(lora_file_name)
        model.to(args.device)
    else:
        logging.info(
            " || ".join(
                [
                    "epoch",
                    "train_acc",
                    "total_loss",
                    "total_cls_loss",
                    "total_diff_loss",
                ]
            )
        )

        for epoch in range(1, args.epochs + 1):
            if epoch > 3 and args.test_label:
                break
            train(
                args,
                task_index,
                epoch,
                model,
                train_loader,
                optimizer,
                scheduler,
                known_class_num,
                accmulate_class_num,
            )
        model.save_lora_parameters(lora_file_name)

    if not args.not_upper_test:
        logging.info(f"  ")
        logging.info(f"====> {task_index} UpperTesting")
        test_acc = compute_current_accuracy(
            args,
            task_index,
            model,
            test_loader,
            known_class_num,
            accmulate_class_num,
        )
        upper_accs.append(test_acc)

    if not args.not_cluster:
        logging.info(f"  ")
        logging.info(f"====> {task_index} Clustering")
        center_file_name = center_file_path(args, task_index)
        centers = clustering(
            args, model, train_loader, center_file_name, known_class_num
        )
        kmeans_centers.append(centers)

    if not args.not_eval:
        logging.info(f"  ")
        logging.info(f"====> {task_index} DomainTesting")
        total_num, mean_acc, tasks_acc = eval_cnn(
            args, model, test_loader, kmeans_centers, known_class_num
        )
        tasks_accs.append(tasks_acc)
        mean_accs.append(mean_acc)
        total_nums.append(total_num)

    known_class_num = accmulate_class_num
    task_end_time = time.time()
    logging.info(f"  ")
    logging.info(f"task {task_index} time: {(task_end_time - task_start_time)/60} m")

end_time = time.time()
logging.info(f"  ")
logging.info(f"  ")
logging.info(f"====> experiment result")
logging.info(f"Total time: {(end_time - start_time)/60/60} h")
logging.info(f"Upper acc: {upper_accs}")
logging.info(f"Tasks acc: {tasks_accs}")
logging.info(f"Mean acc: {mean_accs}")
logging.info(f"Total num: {total_nums}")
