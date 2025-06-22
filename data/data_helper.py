"""
    A unified dataset interface.
"""

import torch.utils.data as data

class Dataset(data.Dataset):
    # data.shape = (B, H, W, C)
    # tensor已经归一化了
    def __init__(self, adv_data, labels):
        self.data = adv_data
        self.labels = labels

    def __getitem__(self, index):
        img = self.data[index]
        img_aug = img
        return img, img_aug, self.labels[index], index

    def __len__(self):
        return len(self.data)