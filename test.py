import logging

import numpy as np
import torch

from lora import set_task_index
from utils import tensor2numpy


@torch.no_grad()
def compute_current_accuracy(args, net, loader, known_class_num, accmulate_class_num):
    net.eval()
    correct, total = 0, 0
    for _, (_, inputs, targets) in enumerate(loader):
        image, label = inputs.to(args.device), targets.to(args.device)
        mask = (label >= known_class_num).nonzero().view(-1)
        if len(mask) == 0:
            continue
        image = torch.index_select(image, 0, mask)
        label = torch.index_select(label, 0, mask)
        with torch.no_grad():
            _, pred = net(image)
        pred[:, 0:known_class_num] = -float("inf")
        pred[:, accmulate_class_num:] = -float("inf")
        _, preds_train = torch.max(pred, dim=1)
        correct += preds_train.eq(label.expand_as(preds_train)).cpu().sum()
        total += len(label)

    test_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
    logging.info(f"upper_acc:{test_acc}, correct:{correct}, total:{total}")
    return test_acc


@torch.no_grad()
def eval_cnn(args, net, loader, kmeans_centers, known_class_num):
    y_pred = []
    y_label = []
    net.eval()
    current_task_num = len(kmeans_centers)
    for _, (_, inputs, targets) in enumerate(loader):
        set_task_index(-1)
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)
        with torch.no_grad():
            features, _ = net(inputs)
            min_distaces = torch.tensor([float("inf")] * features.shape[0]).to(
                args.device
            )
            min_idxs = torch.zeros(features.shape[0]).to(args.device)
            for idx, centers in enumerate(kmeans_centers):
                idxs = torch.full((features.shape[0],), idx, dtype=torch.int32).to(
                    args.device
                )
                for center in centers:
                    distaces = torch.norm(features - center, dim=1)
                    min_idxs = torch.where(distaces < min_distaces, idxs, min_idxs)
                    min_distaces = torch.where(
                        distaces < min_distaces, distaces, min_distaces
                    )
            # logging.info(f"{min_idxs.cpu().detach().numpy()}")
            for idx in range(current_task_num):
                set_task_index(idx)
                # mask = (min_idxs == idx).nonzero().view(-1)
                mask = (targets // 2 == idx).nonzero().view(-1)
                if len(mask) == 0:
                    continue
                image = torch.index_select(inputs, 0, mask)
                label = torch.index_select(targets, 0, mask)
                _, logits = net.forward(image)
                _, pred = torch.max(logits, dim=1)
                y_pred.append(pred.cpu().numpy())
                y_label.append(label.cpu().numpy())

    y_pred = np.concatenate(y_pred)
    y_label = np.concatenate(y_label)

    mean_acc = compute_accurcy(args, y_pred, y_label)

    start = 0
    end = 0
    all_tasks_acc = []
    for idx in range(current_task_num):
        end += args.class_num_per_task_list[idx]
        idxes = np.where(np.logical_and(y_label >= start, y_label < end))[0]
        all_tasks_acc.append(compute_accurcy(args, y_pred[idxes], y_label[idxes]))
        start = end

    idxes = np.where(y_label < known_class_num)[0]
    old_acc = (
        0 if len(idxes) == 0 else compute_accurcy(args, y_pred[idxes], y_label[idxes])
    )

    idxes = np.where(y_label >= known_class_num)[0]
    new_acc = compute_accurcy(args, y_pred[idxes], y_label[idxes])

    logging.info(f"mean_acc: {mean_acc}")
    logging.info(f"all_tasks_acc: {all_tasks_acc}")
    logging.info(f"old_task_acc: {old_acc}")
    logging.info(f"current_task_acc: {new_acc}")
    return len(y_label), mean_acc, all_tasks_acc


def compute_accurcy(args, y_pred, y_label):
    assert len(y_pred) == len(y_label)
    class_num_per_task = args.classes_num // args.tasks_num
    return np.around(
        (y_pred % class_num_per_task == y_label % class_num_per_task).sum()
        * 100
        / len(y_label),
        decimals=2,
    )
