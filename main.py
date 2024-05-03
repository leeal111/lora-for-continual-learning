import argparse
import logging
import time
from os.path import exists
import numpy as np
from data_manager import DataManager
from model import load_vit_train_type
from test import compute_current_accuracy, eval_cnn
from train import clustering, init_optimizer, init_other, train
from utils import center_file_path, init_args, init_logging, weight_file_path
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
# old args
parser.add_argument("--batch_size", type=int, default=50)
# parser.add_argument("--lora_layers_start", type=int, default=0)
# parser.add_argument("--lora_layers_end", type=int, default=11)
parser.add_argument("--saveFea_loraId", type=int, default=0)
# parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--dataset", type=str)  # domainnet
parser.add_argument("--data_path", type=str)
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--num_classes", "-nc", type=int, default=345 * 6)
parser.add_argument("--num_tasks", type=int, default=5)
# parser.add_argument("--init_cls", type=int, default=2)
# parser.add_argument("--increment", type=int, default=2)
parser.add_argument("--train_type", "-tt", type=str, default="lora")
parser.add_argument("--rank", "-r", type=int, default=2)
# parser.add_argument("--gpu", default=3)
parser.add_argument(
    "--task_name",
    nargs="+",
    default=["gaugan", "biggan", "wild", "whichfaceisreal", "san"],
)
parser.add_argument(
    "--class_order", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
)
parser.add_argument("--multiclass", type=int, nargs="+", default=[0, 0, 0, 0, 0])

# new args
parser.add_argument("--pretrain_model_path", type=str)
parser.add_argument("--tasks_lr_T", type=int, nargs="+", default=[120, 48, 125, 24, 6])
parser.add_argument("--class_num_per_task_list", type=int, nargs="+")
parser.add_argument("--gpus", type=int, nargs="+", default=[0])
parser.add_argument("--n_clusters", type=int, default=5)
parser.add_argument("--pretrain_model_name", type=str)
parser.add_argument("--result_path", type=str, default="./results")
parser.add_argument("--raitolossA", type=float, default=0)
parser.add_argument("--raitolossB", type=float, default=0)
parser.add_argument("--loss_ratio_start", type=float, default=1)
parser.add_argument("--loss_ratio_end", type=float, default=1)
parser.add_argument("--ratio_type", type=str, default="linear")

# parser.add_argument("--raitolossTeacher", type=float, default=0)
# parser.add_argument("--enable_load_weight", action="store_true")
# parser.add_argument("--enable_load_center", action="store_true")
# parser.add_argument("--not_upper_test", action="store_true")
# parser.add_argument("--not_cluster", action="store_true")
# parser.add_argument("--not_eval", action="store_true")
# parser.add_argument("--class_sum", action="store_true")

args = parser.parse_args()

init_args(args)

init_logging(args.log_path)
init_other(args)

logging.info(f"experiment settings:")
for arg_name, arg_value in args.__dict__.items():
    logging.info(f"{arg_name} : {str(arg_value)}")
logging.info(f"  ")

# load model
model = load_vit_train_type(args)
model.to(args.device)

# load data_manager
data_manager = DataManager(
    args.dataset, False, args.seed, args.init_cls, args.increment, args
)


start_time = time.time()
known_classes = 0
upper_accs = []
kmeans_centers = []
tasks_accs = []
mean_accs = []
total_nums = []
for task_index in range(args.num_tasks):
    task_start_time = time.time()

    # get class tag range
    current_class_num = args.class_num_per_task_list[task_index]
    total_classes = current_class_num + known_classes
    logging.info(f"  ")
    logging.info(f"====>   ")
    logging.info(
        f"====> {task_index} Task Learn with class range: {known_classes}-{total_classes}"
    )
    logging.info(f"====>   ")
    logging.info(f"  ")

    # load data
    train_dataset = data_manager.get_dataset(
        np.arange(known_classes, total_classes), source="train", mode="train"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_dataset = data_manager.get_dataset(
        np.arange(0, total_classes), source="test", mode="test"
    )  # Core50: 44972
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # 准备训练
    optimizer, scheduler = init_optimizer(args, model, task_index)
    logging.info(f"  ")
    logging.info(f"====> {task_index} Training")
    lora_file_name = weight_file_path(args, task_index)
    if exists(lora_file_name):
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
            train(
                args,
                task_index,
                epoch,
                model,
                train_loader,
                optimizer,
                scheduler,
                known_classes,
                total_classes,
            )
        # model.save_lora_parameters(lora_file_name)

    logging.info(f"  ")
    logging.info(f"====> {task_index} UpperTesting")
    test_acc = compute_current_accuracy(
        args,
        task_index,
        model,
        test_loader,
        known_classes,
        total_classes,
    )
    upper_accs.append(test_acc)

    logging.info(f"  ")
    logging.info(f"====> {task_index} Clustering")
    center_file_name = center_file_path(args, task_index)
    centers = clustering(args, model, train_loader, center_file_name, known_classes)
    kmeans_centers.append(centers)

    logging.info(f"  ")
    logging.info(f"====> {task_index} DomainTesting")
    total_num, mean_acc, tasks_acc = eval_cnn(
        args,
        model,
        test_loader,
        kmeans_centers,
        known_classes,
    )
    tasks_accs.append(tasks_acc)
    mean_accs.append(mean_acc)
    total_nums.append(total_num)

    known_classes = total_classes
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
