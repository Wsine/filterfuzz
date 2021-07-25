# source: https://raw.githubusercontent.com/tomlawrenceuk/GTSRB-Dataloader/master/gtsrb_dataset.py

import torch
import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class GTSRB(Dataset):
    base_folder = 'GTSRB'

    def __init__(self, root_dir, stype, transform=None):
        """
        Args:
            root_dir (string): Directory containing GTSRB folder.
            stype (string): Load train/val/test set.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert stype in ('train', 'val', 'test'), 'Invalid set type'
        self.root_dir = root_dir
        self.stype = stype

        self.sub_directory = 'trainingset' if stype != 'test' else 'testset'
        self.csv_file_name = 'training.csv' if stype != 'test' else 'test.csv'

        csv_file_path = os.path.join(
            self.root_dir, self.base_folder, self.sub_directory, self.csv_file_name)

        self.csv_data = pd.read_csv(csv_file_path)
        if self.stype == 'train':
            val_data = self.csv_data.iloc[:, 0].map(lambda p: int(p.split('/')[1].split('_')[0])).isin((0, 1, 2))
            self.csv_data.drop(self.csv_data[val_data].index, inplace=True)
        elif self.stype == 'val':
            val_data = self.csv_data.iloc[:, 1].map(lambda p: int(p.split('/')[1].split('_')[0])).isin((0, 1, 2))
            self.csv_data = self.csv_data[val_data]

        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.base_folder, self.sub_directory,
                                self.csv_data.iloc[idx, 0])
        img = Image.open(img_path)

        classId = self.csv_data.iloc[idx, 1]

        if self.transform is not None:
            img = self.transform(img)

        return img, classId
