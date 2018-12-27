import cv2
import torch
import numpy as np

class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self,sample):
        img, label, target, seq_len = sample['image'], sample['label'], sample['target'], sample['seq_length']
        h, w = img.shape
        output_h, output_w = self.output_size
        ratio1 = w / output_w
        ratio2 = h / output_h
        if h / ratio1 < output_h:
            img = cv2.resize(img, (output_w, int(h / ratio1)))
        else:
            img = cv2.resize(img, (int(w / ratio2), output_h))
        return {'image': img, 'label': label, 'target': target, 'seq_length': seq_len}


class Padding(object):

    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, sample):
        img, label, target, seq_len = sample['image'], sample['label'], sample['target'], sample['seq_length']
        h, w= img.shape
        output_h, output_w = self.output_size
        assert (h == output_h or w == output_w)
        if h == output_h:
            img = cv2.copyMakeBorder(img, 0, 0, 0, abs(w - output_w), cv2.BORDER_CONSTANT, value=255)
        elif w == output_w:
            img = cv2.copyMakeBorder(img, 0, abs(h - output_h), 0, 0, cv2.BORDER_CONSTANT, value=255)
        return {'image': img, 'label': label, 'target': target, 'seq_length': seq_len}


class ToTensor(object):

    def __call__(self, sample):
        img, label, target, seq_len = sample['image'], sample['label'], sample['target'], sample['seq_length']
        img = np.expand_dims(img, axis = 0)
        img = torch.from_numpy(img).type('torch.DoubleTensor')
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        return {'image': img, 'label': label, 'target': target, 'seq_length': seq_len}
