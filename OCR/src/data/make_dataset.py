import cv2
import os
import re
import glob
from torch.utils.data import Dataset

class IAM_Dataset(Dataset):

    def __init__(self,root_dir,transform=None, set='training'):
        self.root_dir = root_dir
        filenames = sorted(glob.glob(root_dir+"/words/*/*/*.png"))
        labels = []
        self.transform = transform
        self.char_to_index = {}
        self.index_to_char = {}
        self.max_label_length = 0
        self.set = set

        self.char_to_index['BLANK'] = 0
        self.index_to_char[0] = 'BLANK'

        labels_path = os.path.join(root_dir,"words.txt")
        with open(labels_path, 'rb') as f:
            last_index = 1
            for line in f.read().splitlines()[18:]:
                match = re.findall("[^ ]+$", line.decode())[0]
                for char in match:
                    if char not in self.char_to_index:
                        self.char_to_index[char] = last_index
                        self.index_to_char[last_index] = char
                        last_index += 1
                if len(match) > self.max_label_length:
                    self.max_label_length = len(match)
                labels.append(match)

        # NoneType images
        del filenames[4152]
        del filenames[113620]
        del filenames[107039]
        del labels[4152]
        del labels[113620]
        del labels[107039]

        self.filenames = filenames
        self.labels = labels

        assert(self.set in ['training', 'validation'])
        if self.set == 'training':
            training_length = int(len(self.filenames)*0.99)
            self.filenames = self.filenames[:training_length]
            self.labels = self.labels[:training_length]
        else:
            validation_length = int(len(self.filenames) * 0.01)
            d = len(self.filenames)
            self.filenames = self.filenames[d-validation_length:]
            self.labels = self.labels[d-validation_length:]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = cv2.imread(self.filenames[idx],cv2.IMREAD_GRAYSCALE)
        label = self.labels[idx]
        seq_len = len(label)
        sample = {'image': img, 'label': label, 'seq_length': seq_len}

        if self.transform and img is not None:
            sample = self.transform(sample)

        return sample


