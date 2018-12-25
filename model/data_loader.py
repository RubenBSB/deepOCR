import cv2
import numpy as np
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
        self.char_to_index = {}
        self.index_to_char = {}

        labels_path = os.path.join(root_dir,"words.txt")
        with open(labels_path, 'rb') as f:
            last_index = 0
            for line in f.read().splitlines()[18:]:
                match = re.findall("[^ ]+$", line.decode())[0]
                for char in match:
                    if char not in self.char_to_index:
                        self.char_to_index[char] = last_index
                        self.index_to_char[last_index] = char
                        last_index += 1
                self.labels.append(match)
        self.char_to_index['BLANK'] = len(self.char_to_index)
        self.index_to_char[len(self.char_to_index)] = 'BLANK'

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = cv2.imread(self.filenames[idx],cv2.IMREAD_GRAYSCALE)
        label = self.labels[idx]
        target = [np.eye(len(self.char_to_index))[self.char_to_index[char]] for char in label]

        sample = {'image': img, 'label': label, 'target': target}

        if self.transform and img is not None:
            sample = self.transform(sample)

        return sample


