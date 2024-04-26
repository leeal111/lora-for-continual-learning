import logging
from os import makedirs
from os.path import join,basename,splitext
import time

def init_args(args):

    # input verify
    if args.dataset_name is None:
        args.dataset_name = splitext(basename(args.dataset_path))[0]
    if args.pretrain_model_name is None:
        args.pretrain_model_name = splitext(basename(args.pretrain_model_path))[0]
    assert args.classes_num >= args.tasks_num
    if args.tasks_name is not None:
        assert len(args.tasks_name)==args.tasks_num
    if args.tasks_lr_T is not None:
        assert len(args.tasks_lr_T)==args.tasks_num
    if args.class_num_per_task_list is not None:
        assert len(args.class_num_per_task_list)==args.tasks_num
        assert sum(args.class_num_per_task_list)==args.classes_num
    else:
        class_num_per_task = args.classes_num // args.tasks_num
        if class_num_per_task*args.tasks_num!=args.classes_num:
            class_num_per_task+=1
        args.class_num_per_task_list = []
        while sum(args.class_num_per_task_list) + class_num_per_task < args.classes_num:
            args.class_num_per_task_list.append(class_num_per_task)
        offset = args.classes_num - sum(args.class_num_per_task_list)
        args.class_num_per_task_list.append(offset)

    # make output diretory
    makedirs(args.result_path)
    args.log_path=join(args.result_path,f"log")
    args.weight_path=join(args.result_path,f"weight")
    args.center_path=join(args.result_path,f"center")
    makedirs(args.log_path, exist_ok=True)
    makedirs(args.weight_path, exist_ok=True)
    makedirs(args.weight_path, exist_ok=True)

    # other
    init_logging(args.log_path)
    

def init_logging(result_path):
    logger = logging.getLogger("")
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s-%(levelname)s] %(message)s",
            datefmt="%y-%m-%d %H:%M:%S",
            filename=join(result_path, f"{time.strftime("%Y%m%d_%H%M%S")}.log"),
            filemode="w",
        )
        console = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s-%(levelname)s] %(message)s")
        console.setFormatter(formatter)
        logger.addHandler(console)
    



