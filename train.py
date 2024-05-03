import logging
import math
import os
import random

import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.optim as optim
from lora import set_task_index
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
    args.scaler = GradScaler()


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
    # torch.use_deterministic_algorithms(True)
    torch.use_deterministic_algorithms(False)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def train(
    args,
    task_index,
    epoch,
    net,
    loader,
    optimizer,
    scheduler,
    known_classes,
    total_classes,
):
    loss_func = nn.CrossEntropyLoss().to(args.device)
    total_cls_loss = 0.0
    total_diff_loss = 0.0
    total_loss = 0.0
    correct, total = 0, 0
    net.train()
    for i, data in enumerate(loader):
        _, image, label = data
        image, label = image.to(args.device), label.to(args.device)
        mask = (label >= known_classes).nonzero().view(-1)
        image = torch.index_select(image, 0, mask)
        label = torch.index_select(label, 0, mask)
        loss = 0.0
        optimizer.zero_grad()
        with autocast(enabled=True):
            set_task_index(task_index)
            _, pred = net.forward(image)
            pred[:, 0:known_classes] = -float("inf")
            pred[:, total_classes:] = -float("inf")
            cls_loss = loss_func(pred, label)

            # teacher_loss = 0.0
            # if not math.isclose(args.raitolossTeacher, 0):
            #     if task_index != 0:
            #         set_task_index(task_index - 1)
            #         fea, _ = net(images)
            #         teacher_loss = (
            #             torch.norm(cur_fea - fea, p=2) * args.raitolossTeacher
            #         )

            diff_loss = dif_loss(args, net, task_index)  # + teacher_loss
            loss = cls_loss + diff_loss

        args.scaler.scale(loss).backward()
        args.scaler.step(optimizer)
        args.scaler.update()
        scheduler.step()

        total_cls_loss += cls_loss
        total_diff_loss += diff_loss
        total_loss += loss.item()

        _, preds_train = torch.max(pred, dim=1)
        correct += preds_train.eq(label.expand_as(preds_train)).cpu().sum()
        total += len(label)

    total_loss /= len(loader)
    total_cls_loss /= len(loader)
    total_diff_loss /= len(loader)

    train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    logging.info(
        f"{epoch:03} {train_acc:.3f} {total_loss:.3f} {total_cls_loss:.3f} {total_diff_loss:.3f}"
    )


def dif_loss(args, net, task_index):
    diff_loss = 0.0
    if task_index == 0:
        return 0
    else:
        for lora_idx in range(len(net.lora_layer)):
            layer1_index = lora_idx * args.num_tasks * 2 + (task_index - 1) * 2
            layer2_index = lora_idx * args.num_tasks * 2 + (task_index) * 2

            diff_loss += (
                inter_radio(args, lora_idx, len(net.lora_layer))
                * args.raitolossA
                * layer_l2_diff(net.w_As[layer1_index], net.w_As[layer2_index])
            )
            diff_loss += (
                inter_radio(args, lora_idx, len(net.lora_layer))
                * args.raitolossA
                * layer_l2_diff(net.w_As[layer1_index + 1], net.w_As[layer2_index + 1])
            )
            diff_loss += (
                inter_radio(args, lora_idx, len(net.lora_layer))
                * args.raitolossB
                * layer_l2_diff(net.w_Bs[layer1_index], net.w_Bs[layer2_index])
            )
            diff_loss += (
                inter_radio(args, lora_idx, len(net.lora_layer))
                * args.raitolossB
                * layer_l2_diff(net.w_Bs[layer1_index + 1], net.w_Bs[layer2_index + 1])
            )

    return diff_loss


def inter_radio(args, index, lora_num):
    if args.ratio_type == "linear":
        if math.isclose(args.loss_ratio_end, args.loss_ratio_start):
            return args.loss_ratio_end
        else:
            return (
                index / lora_num * (args.loss_ratio_end - args.loss_ratio_start)
                + args.loss_ratio_start
            )


def layer_l2_diff(layer1, layer2):
    # 获取两个线性层之间的参数
    parameters1 = torch.cat([param.view(-1) for param in layer1.parameters()])
    parameters2 = torch.cat([param.view(-1) for param in layer2.parameters()])

    # 计算参数差异的L2范数
    return torch.norm(parameters1 - parameters2, p=2)


def clustering(args, net, loader, path, known_class_num):
    if exists(path):
        logging.info(f"load center data")
        centers = np.load(path)
    else:
        logging.info(f"search center data")
        features = []
        for i, (_, inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            mask = (targets >= known_class_num).nonzero().view(-1)
            inputs = torch.index_select(inputs, 0, mask)
            with torch.no_grad():
                set_task_index(args.saveFea_loraId)
                feature, _ = net(inputs)
                features.append(feature.cpu())
        features = torch.cat(features, 0).cpu().detach().numpy()
        clustering = KMeans(n_clusters=args.n_clusters, random_state=0, n_init=10).fit(
            features
        )
        centers = clustering.cluster_centers_
        # np.save(path, centers)
    return torch.tensor(centers).to(args.device)
