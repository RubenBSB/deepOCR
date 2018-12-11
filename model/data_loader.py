import cv2
import glob
from torch.utils.data import Dataset, Dataloader


class IAM_Dataset(Dataset):

    def __init__(self,labels,transform):
        self.root = "words/*/*/*.png"
        self.filenames = glob.glob(root)
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = cv2.imread(self.filenames[idx])
        img = self.transform(img)
        return img, self.labels[idx]
