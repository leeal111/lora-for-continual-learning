import logging

import numpy as np
import torch

from lora import set_task_index
from utils import tensor2numpy


@torch.no_grad()
def compute_current_accuracy(
    args, task_index, net, loader, known_class_num, accmulate_class_num
):
    net.eval()
    correct, total = 0, 0
    for _, (_, inputs, targets) in enumerate(loader):
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)
        # mask = (targets >= known_class_num).nonzero().view(-1)
        # if len(mask) == 0:
        #     continue
        # inputs = torch.index_select(inputs, 0, mask)
        # targets = torch.index_select(targets, 0, mask)
        with torch.no_grad():
            set_task_index(task_index)
            _, logits = net.forward(inputs)
        logits[:, 0:known_class_num] = -float("inf")
        logits[:, accmulate_class_num:] = -float("inf")
        predicts = torch.max(logits, dim=1)[1]
        correct += (
            (predicts % args.increment).cpu() == (targets % args.increment).cpu()
        ).sum()
        total += len(targets)

    test_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
    logging.info(f"upper_acc:{test_acc}, correct:{correct}, total:{total}")
    return test_acc


# @torch.no_grad()
# def _eval_cnn(args, net, loader, kmeans_centers):
#     net.eval()
#     y_pred, y_true = [], []
#     topk = 2
#     pick_id_list = []
#     for _, (_, inputs, targets) in enumerate(loader):
#         inputs = inputs.to(args.device)
#         targets = targets.to(args.device)

#         with torch.no_grad():
#             set_task_index(args.saveFea_loraId)
#             feature, logit = net(inputs)  # mom=0, mom=2
#             # feature, logit = net(inputs,task_id=0,mom=0,training=False)
#             # 本来应该只用ViT，不加lora，来抽特征。为了写代码方便，这里改成统一用vit+lora0
#             # features = features / features.norm(dim=-1, keepdim=True)  # 这句sinet好像没有

#             taskselection = []
#             for task_centers in kmeans_centers:
#                 tmpcentersbatch = []
#                 for center in task_centers:
#                     tmpcentersbatch.append((((feature - center) ** 2) ** 0.5).sum(1))
#                 taskselection.append(torch.vstack(tmpcentersbatch).min(0)[0])

#             selection = torch.vstack(taskselection).min(0)[1]
#             # print(f"selection:{selection}")

#             "---------- 改成按batch来选id吧 ------------"
#             # 使用torch.mode()函数找到张量中出现最多的值和计数
#             pick_id, count = torch.mode(selection)
#             set_task_index(pick_id)
#             _, logits = net.forward(inputs)
#             # print(f"--- pick_id:{pick_id}")
#             pick_id_list.append(pick_id.item())

#             # mask_classes(logits, pick_id, cfg.increment)

#         predicts = torch.topk(logits, k=topk, dim=1, largest=True, sorted=True)[
#             1
#         ]  # [bs, topk]
#         y_pred.append(predicts.cpu().numpy())
#         y_true.append(targets.cpu().numpy())

#     print(f"**** pick_id_list:{pick_id_list}")
#     return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]


# def accuracy_domain(y_pred, y_true, nb_old, increment, class_num):
#     assert len(y_pred) == len(y_true), "Data length error."
#     all_acc = {}
#     all_acc["total"] = np.around(
#         (y_pred % class_num == y_true % class_num).sum() * 100 / len(y_true), decimals=2
#     )

#     # Grouped accuracy
#     for class_id in range(0, np.max(y_true), increment):
#         idxes = np.where(
#             np.logical_and(y_true >= class_id, y_true < class_id + increment)
#         )[0]
#         label = "{}-{}".format(
#             str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
#         )
#         all_acc[label] = np.around(
#             ((y_pred[idxes] % class_num) == (y_true[idxes] % class_num)).sum()
#             * 100
#             / len(idxes),
#             decimals=2,
#         )

#     # Old accuracy
#     idxes = np.where(y_true < nb_old)[0]
#     all_acc["old"] = (
#         0
#         if len(idxes) == 0
#         else np.around(
#             ((y_pred[idxes] % class_num) == (y_true[idxes] % class_num)).sum()
#             * 100
#             / len(idxes),
#             decimals=2,
#         )
#     )

#     # New accuracy
#     idxes = np.where(y_true >= nb_old)[0]
#     all_acc["new"] = np.around(
#         ((y_pred[idxes] % class_num) == (y_true[idxes] % class_num)).sum()
#         * 100
#         / len(idxes),
#         decimals=2,
#     )

#     return all_acc


@torch.no_grad()
def eval_cnn(args, net, loader, kmeans_centers, known_class_num):
    net.eval()
    y_pred = []
    y_label = []
    pick_id_list = []
    current_task_num = len(kmeans_centers)
    for _, (_, inputs, targets) in enumerate(loader):
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)
        with torch.no_grad():
            set_task_index(args.saveFea_loraId)
            features, _ = net(inputs)

            taskselection = []
            for task_centers in kmeans_centers:
                tmpcentersbatch = []
                for center in task_centers:
                    tmpcentersbatch.append((((features - center) ** 2) ** 0.5).sum(1))
                taskselection.append(torch.vstack(tmpcentersbatch).min(0)[0])
            selection = torch.vstack(taskselection).min(0)[1]

            # min_distaces = torch.tensor([float("inf")] * features.shape[0]).to(
            #     args.device
            # )
            # selection = torch.zeros(features.shape[0]).to(args.device)
            # for idx, centers in enumerate(kmeans_centers):
            #     idxs = torch.full((features.shape[0],), idx, dtype=torch.int32).to(
            #         args.device
            #     )
            #     for center in centers:
            #         distaces = torch.norm(features - center, dim=1)
            #         selection = torch.where(distaces < min_distaces, idxs, selection)
            #         min_distaces = torch.where(
            #             distaces < min_distaces, distaces, min_distaces
            #         )

            select_task_index, _ = torch.mode(selection)
            pick_id_list.append(int(select_task_index.item()))
            set_task_index(int(select_task_index.item()))
            _, logits = net.forward(inputs)
            _, predicts = torch.max(logits, dim=1)
            y_pred.append(predicts.cpu().numpy())
            y_label.append(targets.cpu().numpy())

            # for idx in range(current_task_num):
            #     set_task_index(idx)
            #     mask = (min_idxs == idx).nonzero().view(-1)
            #     # mask = (targets//2 == idx).nonzero().view(-1)
            #     if len(mask) == 0:
            #         continue
            #     image = torch.index_select(inputs, 0, mask)
            #     label = torch.index_select(targets, 0, mask)
            #     _, logits = net.forward(image)
            #     if not args.class_sum:
            #         logits[:, 0 : sum(args.class_num_per_task_list[0:idx])] = -float(
            #             "inf"
            #         )
            #         logits[:, sum(args.class_num_per_task_list[0 : idx + 1]) :] = (
            #             -float("inf")
            #         )
            #     _, pred = torch.max(logits, dim=1)
            #     y_pred.append(pred.cpu().numpy())
            #     y_label.append(label.cpu().numpy())

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
    class_num_per_task = args.num_classes // args.num_tasks
    return np.around(
        (y_pred % class_num_per_task == y_label % class_num_per_task).sum()
        * 100
        / len(y_label),
        decimals=2,
    )
