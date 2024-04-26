import os
import numpy as np
from torchvision import transforms


def split_images_labels(datas):
    images = []
    labels = []
    for item in datas:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)


class iGanFake(object):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self):
        pass

    def init_data(self, args):
        train_dataset = self.get_dataset(args, "train")
        test_dataset = self.get_dataset(args, "val")
        self.train_data, self.train_targets = split_images_labels(train_dataset)
        self.test_data, self.test_targets = split_images_labels(test_dataset)

    def get_dataset(self, args, type_name):
        dataset = []
        for index, name in enumerate(args.tasks_name):
            root = os.path.join(args.dataset_path, name, type_name)
            for filename in os.listdir(os.path.join(root, "0_real")):
                dataset.append((os.path.join(root, "0_real", filename), 0 + 2 * index))
            for filename in os.listdir(os.path.join(root, "1_fake")):
                dataset.append((os.path.join(root, "1_fake", filename), 1 + 2 * index))
        return dataset
