import logging
from os import makedirs
from os.path import join, basename, splitext
import time

from lora import get_task_index


def init_args(args):

    # input verify
    if args.dataset_name is None:
        args.dataset_name = splitext(basename(args.dataset_path))[0]
    if args.pretrain_model_name is None:
        args.pretrain_model_name = splitext(basename(args.pretrain_model_path))[0]
    assert args.classes_num >= args.tasks_num
    if args.tasks_name is not None:
        assert len(args.tasks_name) == args.tasks_num
    if args.tasks_lr_T is not None:
        assert len(args.tasks_lr_T) == args.tasks_num
    if args.class_num_per_task_list is not None:
        assert len(args.class_num_per_task_list) == args.tasks_num
        assert sum(args.class_num_per_task_list) == args.classes_num
    else:
        class_num_per_task = args.classes_num // args.tasks_num
        if class_num_per_task * args.tasks_num != args.classes_num:
            class_num_per_task += 1
        args.class_num_per_task_list = []
        while sum(args.class_num_per_task_list) + class_num_per_task < args.classes_num:
            args.class_num_per_task_list.append(class_num_per_task)
        offset = args.classes_num - sum(args.class_num_per_task_list)
        args.class_num_per_task_list.append(offset)

    # make output diretory
    args.log_path = join(args.result_path, f"log")
    args.weight_path = join(args.result_path, f"weight")
    args.center_path = join(args.result_path, f"center")
    makedirs(args.log_path, exist_ok=True)
    makedirs(args.weight_path, exist_ok=True)
    makedirs(args.center_path, exist_ok=True)


def init_logging(result_path, file_name):
    logger = logging.getLogger("")
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s-%(levelname)s] %(message)s",
            datefmt="%y-%m-%d %H:%M:%S",
            filename=join(result_path, f"{file_name}.log"),
            filemode="w",
        )
        console = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s-%(levelname)s] %(message)s")
        console.setFormatter(formatter)
        logger.addHandler(console)


def print_trainable_size(name, param):
    num_params = sum(p.numel() for p in param if p.requires_grad)
    logging.info(f"{name} trainable size: {num_params / 2**20:.4f}M")


param_list = [
    "dataset_name",
    "train_type",
    "tasks_num",
    "rank",
    "seed",
    "epochs",
    "tasks_lr_T",
    "raitolossA",
    "raitolossB",
]


def weight_file_path(args, task_index):

    param_str = get_param_str(args, param_list)
    unique_file_str = f"{param_str}_{task_index}.safetensors"
    lora_file_name = join(args.weight_path, "lora_" + unique_file_str)
    return lora_file_name


def get_param_str(args, param_list):
    param_values = []
    for arg_name, arg_value in args.__dict__.items():
        if arg_name in param_list:
            param_values.append(str(arg_value))
    assert len(param_list) == len(param_values)
    param_str = "_".join(param_values)
    return param_str


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def center_file_path(args, task_index):
    infer_task_index = get_task_index()
    param_str = get_param_str(args, param_list)
    if infer_task_index == -1:
        unique_file_str = f"{infer_task_index}_{args.classes_num}_{task_index}.npy"
    else:
        unique_file_str = (
            f"{infer_task_index}_{args.classes_num}_{param_str}_{task_index}.npy"
        )
    file_name = join(args.center_path, unique_file_str)
    return file_name


def argv_str(argv):
    exlusion_list = ["--result_path", "--gpus", "--workers_num"]
    add_flag = True
    new_list = []
    for item in argv:
        if item.startswith("--") and item not in exlusion_list:
            add_flag = True
        if item in exlusion_list:
            add_flag = False
        if add_flag:
            new_list.append(item)
    return "_".join(new_list).replace("-", "")
