import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split

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
        labels = [y] * len(imgs)
        return imgs, trfm, labels


def load_dataset(opt, set_type='test'):
    mean = (0.3337, 0.3064, 0.3171)
    std = (0.2672, 0.2564, 0.2629)

    transformers = [
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize(mean, std)
    ]

    dataset = GTSRB(
        root_dir=opt.data_dir, stype=set_type,
        transform=T.Compose(transformers)
    )

    return dataset


def _custom_collate(batch):
    inputs, descs, targets = [], [], []
    for imgs, trfm, labels in batch:
        inputs += imgs
        targets += labels
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

