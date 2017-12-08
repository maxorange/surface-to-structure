import numpy as np
import glob
import os
import cv2
from scipy import ndimage

class Dataset(object):

    def __init__(self, args):
        self.n_gpus = args.n_gpus
        self.index_in_epoch = 0
        self.examples = np.array(glob.glob(args.dataset_path))
        self.test_data = np.array(glob.glob(args.dataset_path.replace('train', 'test')))
        self.num_examples = len(self.examples)
        np.random.shuffle(self.examples)

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.num_examples:
            np.random.shuffle(self.examples)
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_examples
        end = self.index_in_epoch
        return self.read_data(start, end)

    def read_x(self, filename, shift):
        original = cv2.imread(filename, 2)
        original = ndimage.shift(original, shift)
        original = np.expand_dims(original, -1)
        return original.astype(np.float32) / 32767.5 - 1

    def read_y(self, filename, shift):
        masked = cv2.imread(filename, 0)
        masked = ndimage.shift(masked, shift)
        masked = np.expand_dims(masked, -1)
        masked = masked.astype(np.float32) / 255
        return masked #+ masked * np.random.normal(0, 0.2, masked.shape)

    def split(self, data):
        return np.split(data, self.n_gpus)

class VHD(Dataset):

    def __init__(self, args):
        super(VHD, self).__init__(args)

    def read_data(self, start, end):
        x = []
        y = []
        for filename in self.examples[start:end]:
            shift = np.random.randint(-60, 60, size=2)
            x.append(self.read_x(filename.replace('input', 'output'), shift))
            y.append(self.read_y(filename, shift))
        x = self.split(np.array(x))
        y = self.split(np.array(y))
        return x, y

    def read_test_data(self, batch_size):
        x = []
        y = []
        for filename in self.test_data[:batch_size]:
            shift = (0, 0)
            x.append(self.read_x(filename.replace('input', 'output'), shift))
            y.append(self.read_y(filename, shift))
        x = self.split(np.array(x))
        y = self.split(np.array(y))
        return x, y
