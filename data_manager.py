import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from dataset import iGanFake
from torch.utils.data import DataLoader


class DataManager(object):
    def __init__(self, args):

        name = args.dataset_name.lower()
        if name == "cddb":
            idata = iGanFake()
        else:
            raise NotImplementedError(f"Unknown dataset {args.dataset_name}.")

        idata.init_data(args)

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

    def get_dataloader(self, args, indices, source, mode):

        dataset = self.get_dataset(indices, source, mode)
        if source == "train":
            return DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers_num,
            )
        elif source == "test":
            return DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers_num,
            )
        else:
            raise ValueError("Unknown data source {}.".format(source))

    def get_dataset(self, indices, source, mode):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        datas, targets = [], []
        for idx in indices:
            indexs = np.where(np.logical_and(y >= idx, y < idx + 1))[0]
            class_data, class_target = x[indexs], y[indexs]
            datas.append(class_data)
            targets.append(class_target)

        datas, targets = np.concatenate(datas), np.concatenate(targets)
        return DummyDataset(datas, targets, trsf, self.use_path)


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
