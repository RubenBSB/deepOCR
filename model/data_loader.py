import cv2
import pandas as pd
import os
import re
import glob
from torch.utils.data import Dataset


class IAM_Dataset(Dataset):

    def __init__(self,root_dir,transform=None):
        self.root_dir = root_dir
        self.filenames = sorted(glob.glob(root_dir+"/words/*/*/*.png"))
        self.labels = []
        self.transform = transform

        labels_path = os.path.join(root_dir,"words.txt")
        with open(labels_path, 'rb') as f:
            for line in f.read().splitlines()[18:]:
                match = re.findall("[^ ]+$", line.decode())[0]
                self.labels.append(match)


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = cv2.imread(self.filenames[idx],cv2.IMREAD_GRAYSCALE)
        sample = {'image': img, 'label': self.labels[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


