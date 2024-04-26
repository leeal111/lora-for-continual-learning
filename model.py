import logging
import torch
from base_vit import ViT
from lora import LoRA_ViT
from torch import nn

from utils import print_trainable_size


def load_vit(args):
    model = ViT(args.pretrain_model_name)
    model.load_state_dict(torch.load(args.pretrain_model_path))
    print_trainable_size("vit", model.parameters())
    return model


def load_vit_train_type(args):
    model = load_vit(args)
    if args.train_type == "lora":
        model = LoRA_ViT(
            model,
            r=args.rank,
            num_classes=args.classes_num,
            num_tasks=args.tasks_num,
        )
    elif args.train_type == "full":
        model.fc = nn.Linear(model.fc.in_features, args.classes_num)
    elif args.train_type == "linear":
        model.fc = nn.Linear(model.fc.in_features, args.classes_num)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        logging.info("Wrong training type")
        exit()
    print_trainable_size(f"vit_{args.train_type}", model.parameters())
    return model



