import logging
import os
import random

import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.optim as optim
from utils import print_trainable_size, tensor2numpy
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp.grad_scaler import GradScaler
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
from os.path import exists


def init_other(args):
    set_random_seed(args.seed)
    args.device = torch.device(
        f"cuda:{args.gpus[0]}" if torch.cuda.is_available() else "cpu"
    )


def init_optimizer(args, net, task_index):

    # 遍历模型的参数并筛选出满足条件的参数进行优化
    params_to_opt = []
    for name, param in net.named_parameters():
        if param.requires_grad:
            if "fc.weight" in name:
                params_to_opt.append(param)
            if "fc.bias" in name:
                params_to_opt.append(param)
            if "transformer.blocks" in name and name.endswith(".weight"):
                items = name.split(".")
                value = -1
                for item in items:
                    try:
                        value = int(item)
                    except ValueError:
                        continue
                if task_index == value:
                    params_to_opt.append(param)

    print_trainable_size("current task", params_to_opt)
    optimizer = optim.Adam(params_to_opt, lr=args.lr)
    scheduler = CosineAnnealingLR(
        optimizer, args.epochs * args.tasks_lr_T[task_index], 1e-6
    )
    return optimizer, scheduler


def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def train(
    args, epoch, net, loader, optimizer, scheduler, known_class_num, accmulate_class_num
):
    scaler = GradScaler()
    loss_func = nn.CrossEntropyLoss().to(args.device)
    total_loss = 0.0
    correct, total = 0, 0
    this_lr = scheduler.get_last_lr()[0]
    net.train()
    for _, datas in enumerate(loader):
        _, images, labels = datas
        images, labels = images.to(args.device), labels.to(args.device)

        loss = 0.0
        optimizer.zero_grad()
        with autocast(enabled=True):
            _, pred = net(images)
            pred[:, 0:known_class_num] = -float("inf")
            pred[:, accmulate_class_num:] = -float("inf")
            cls_loss = loss_func(pred, labels)
            # A_loss=net
            loss = cls_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        _, preds_train = torch.max(pred, dim=1)
        correct += preds_train.eq(labels.expand_as(preds_train)).cpu().sum()
        total += len(labels)

    total_loss /= len(loader)

    train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    logging.info(
        f"{epoch:03}  {total_loss:.3f}  {train_acc:.3f}  {correct:06}  {total:06}  {this_lr:.3e}"
    )


@torch.no_grad()
def clustering(args, net, loader, path):
    if exists(path):
        logging.info(f"load center data")
        centers = np.load(path)
    else:
        logging.info(f"search center data")
        net.eval()
        features = []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            with torch.no_grad():
                feature, _ = net(inputs)
            features.append(feature.cpu())
        features = torch.cat(features, 0).cpu().detach().numpy()
        clustering = KMeans(
            n_clusters=args.n_clusters, random_state=args.seed, n_init=10
        ).fit(features)
        centers = clustering.cluster_centers_
        np.save(path, centers)
    return torch.tensor(centers).to(args.device)
