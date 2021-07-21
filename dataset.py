import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from datasets.gtsrb import GTSRB


class PostTransformDataset(Dataset):
    def __init__(self, dataset, mutators):
        self.dataset = dataset
        self.mutators = mutators

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        q = self.mutators[idx]
        imgs = []
        trfm = []
        for t in q:
            for pic, mut in t(x, idx):
                imgs.append(pic)
                trfm.append(mut)
        labels = [y] * len(q)
        return imgs, trfm, labels


def load_dataset(opt):
    if opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == "cifar100":
        dataset = torchvision.datasets.CIFAR100
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'gtsrb':
        dataset = GTSRB
        mean = (0.3337, 0.3064, 0.3171)
        std = (0.2672, 0.2564, 0.2629)
    else:
        raise ValueError("Invalid dataset value")

    common_transformers = [
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize(mean, std)
    ]

    testset = dataset(
        root=opt.data_dir, train=False, download=True,
        transform=T.Compose(common_transformers)
    )

    return testset


def _custom_collate(batch):
    inputs, descs, targets = [], [], []
    for imgs, trfm, labels in batch:
        for img, label in zip(imgs, labels):
            inputs.append(img)
            targets.append(label)
        descs += trfm
    return torch.stack(inputs), descs, torch.LongTensor(targets)


def generate_test_dataset(opt, testset, mutators=None):
    if mutators is not None:
        testset = PostTransformDataset(testset, mutators)
    testloader = DataLoader(
        testset, batch_size=opt.batch_size, shuffle=False, num_workers=32,
        collate_fn=_custom_collate if mutators is not None else None
    )
    return testloader

