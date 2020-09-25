import os
import random
import numpy as np

from torch.utils.data.dataset import Dataset

# Dataloder for all Moving-MNIST datasets (binary and colored)
class MNIST_Dataset(Dataset):

    def __init__(self, params):
        # parameters of the dataset 
        path = params['path']
        assert os.path.exists(path), "The file does not exist."

        self.num_frames  = params['num_frames']
        self.num_samples = params.get('num_samples', None)

        self.random_crop = params.get('random_crop', False) 

        self.img_height   = params.get('height',  64)
        self.img_width    = params.get('width',   64)
        self.img_channels = params.get('channels', 3)

        self.data = np.float32(np.load(path)["data"] / 255.0)
        self.data_samples = self.data.shape[0]
        self.data_frames  = self.data.shape[1]

    def __getitem__(self, index):
        start = random.randint(0, self.data_frames - 
            self.num_frames) if self.random_crop else 0

        data  = self.data[index, start : start + self.num_frames]

        return data 

    def __len__(self):
        return len(self.data_samples) if self.num_samples is None \
            else min(self.data_samples,  self.num_samples)