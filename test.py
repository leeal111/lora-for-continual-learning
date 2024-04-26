import logging

import numpy as np
import torch

from utils import tensor2numpy


@torch.no_grad()
def compute_current_accuracy(args, net, loader, known_class_num, accmulate_class_num):
    logging.info(f"\n\n====> Testing")
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
