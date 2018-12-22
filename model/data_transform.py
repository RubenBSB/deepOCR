import cv2
import torch

class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self,sample):
        img, label = sample['image'], sample['label']
        if img is not None:
            h,w = img.shape
            output_h, output_w = self.output_size
            ratio1 = w / output_w
            ratio2 = h / output_h
            if h / ratio1 < output_h:
                img = cv2.resize(img, (output_w, int(h / ratio1)))
            else:
                img = cv2.resize(img, (int(w / ratio2), output_h))
            return {'image': img, 'label': label}

class Padding(object):

    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        if img is not None:
            h, w = img.shape
            output_h, output_w = self.output_size
            assert(h == output_h or w == output_w)
            if h == output_h:
                img = cv2.copyMakeBorder(img,0,0,0,abs(h-output_w),cv2.BORDER_CONSTANT,value=255)
            elif w == output_w:
                img = cv2.copyMakeBorder(img,0,abs(h-output_h),0,0,cv2.BORDER_CONSTANT,value=255)
            return {'image': img, 'label': label}

class ToTensor(object):

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        if img is not None:
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            img = img.transpose((1, 0))
            return {'image': torch.from_numpy(img),
                    'landmarks': label}