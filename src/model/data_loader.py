import cv2
import numpy as np
import os
import re
import glob
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch

class IAM_Dataset(Dataset):

    def __init__(self,root_dir,transform=None, set='training'):
        self.root_dir = root_dir
        self.filenames = sorted(glob.glob(root_dir+"/words/*/*/*.png"))
        self.labels = []
        self.transform = transform
        self.char_to_index = {}
        self.index_to_char = {}
        self.max_label_length = 0
        self.set = set

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
                if len(match) > self.max_label_length:
                    self.max_label_length = len(match)
                self.labels.append(match)

        self.char_to_index['BLANK'] = len(self.char_to_index)
        self.index_to_char[len(self.char_to_index)] = 'BLANK'

        assert(self.set in ['training', 'validation'])
        if self.set == 'training':
            training_length = int(len(self.filenames)*0.99)
            self.filenames = self.filenames[:training_length]
            self.labels = self.labels[:training_length]
        else:
            validation_length = int(len(self.filenames) * 0.01)
            self.filenames = self.filenames[len(self.filenames)-validation_length:]
            self.labels = self.labels[len(self.filenames)-validation_length:]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = cv2.imread(self.filenames[idx],cv2.IMREAD_GRAYSCALE)
        label = self.labels[idx]
        target = torch.tensor([self.char_to_index[char] for char in label])
        seq_len = len(target)
        padding_size = self.max_label_length - len(target)
        target = F.pad(target,(0,padding_size),"constant",78)
        sample = {'image': img, 'label': label, 'target': target, 'seq_length': seq_len}

        if self.transform and img is not None:
            sample = self.transform(sample)

        return sample


