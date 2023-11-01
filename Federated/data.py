import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import Subset
from abc import ABC
import constants
import torch

class Data(ABC, Dataset):
    """
    Constructs a subset of EMNIST dataset from a pickle file;
    expects pickle file to store list of indices
    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets
    Methods
    -------
    __init__
    __len__
    __getitem__
    """

    def __init__(self, X, y, green_idxs, red_idxs):
        """
        :param path: path to .pkl file; expected to store list of indices
        :param emnist_data: EMNIST dataset inputs
        :param emnist_targets: EMNIST dataset labels
        :param transform:
        """
        self.X = torch.Tensor(X)
        self.y = torch.LongTensor(y)
        self.green_idxs = green_idxs
        self.red_idxs = red_idxs
        self.random_explore = torch.zeros(len(X),dtype=bool)
        self.strategy_explore = torch.zeros(len(X),dtype=bool)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        raise NotImplementedError()

    #Sample num samples based on random sampling
    def random_sampler(self, num):
        idxs = np.random.choice(np.arange(len(self.X)), 
                               num, replace=False)
        return Subset(self, idxs), idxs
    
    def random_sampler_green(self, num):
        idxs = np.random.choice(self.green_idxs, 
                               num, replace=False)
        return Subset(self, idxs), idxs
    
    def green_red_sampler(self, num, green_ratio=None):
        assert green_ratio != None, "Pass a valid green ration if you use this function"
        green_num = int(num * green_ratio)
        red_num = num - green_num

        idxs = np.random.choice(self.green_idxs, 
                               green_num, replace=False)
        idxs = np.concatenate((idxs, np.random.choice(self.red_idxs, red_num, replace=False)), axis=0)
        assert len(idxs) == num, "We should exhaust the sampling budget in any case"
        return Subset(self, idxs), idxs

    
    #Sample samples_num samples based on probability distribution
    def probability_sampler(self, num, probabilities):
        idxs = np.random.choice(np.arange(len(self.X)), num, 
                               replace=False, p =probabilities)
        return Subset(self, idxs), idxs
    
    #Sample top num samples based on samples_values
    def top_n_sampler(self, num, samples_values):
        idxs = sorted(range(len(samples_values)), reverse= True, 
                     key=lambda x: samples_values[x])[:num]
        return Subset(self, idxs), idxs
    
    #Sample bottom num based on samples_values
    def bottom_n_sampler(self, num, samples_values):
        idxs = sorted(range(len(samples_values)), reverse= False, 
                     key=lambda x: samples_values[x])[:num]
        return Subset(self, idxs), idxs
    
    def all_sampler(self):
        return Subset(self, range(len(self))), torch.arange(len(self.y))
    
    def green_sampler_full(self):
        return Subset(self, self.green_idxs), self.green_idxs
    
    def get_green_red_ratio(self, idxs) -> dict:
        green = list(set(idxs).intersection(set(self.green_idxs)))
        red = list(set(idxs).intersection(set(self.red_idxs)))
        return {"green" : len(green) / len(idxs), "red" : len(red)/len(idxs),
                "green_idxs" : green, "red_idxs" : red}
    
    def mask_sampler(self, mask):
        idxs = np.where(mask==1)[0]
        if len(idxs) == 0:
            print("No valuable sample found and hence return 32 random samples")
            return self.random_sampler(10)
        return Subset(self, idxs), idxs

    def index_sampler(self, idxs):
        return Subset(self, idxs), idxs

class FLOWERSDataset(Data):
    def __init__(self, X, y, green_idxs, red_idxs):
        super().__init__(X, y, green_idxs, red_idxs)
        self.transform = constants.FLOWERS_TRANSFORM

    def __getitem__(self, index):
        img, target = self.X[index], self.y[index]

        img = Image.fromarray((img.numpy()* 255).astype(np.uint8))

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index


class SVHNDataset(Data):
    def __init__(self, X, y, green_idxs, red_idxs):
        super().__init__(X, y, green_idxs, red_idxs)
        self.transform = constants.SVHN_TRANSFORM

    def __getitem__(self, index):
        img, target = self.X[index], self.y[index]

        '''img = Image.fromarray((img.numpy()* 255).astype(np.uint8))

        if self.transform is not None:
            img = self.transform(img)'''

        return img, target, index

class CIFARDataset(Data):
    def __init__(self, X, y, green_idxs, red_idxs):
        super().__init__(X, y, green_idxs, red_idxs)
        self.transform = constants.CIFAR_TRANSFORM

    def __getitem__(self, index):
        img, target = self.X[index], self.y[index]

        img = Image.fromarray((img.numpy()* 255).astype(np.uint8))

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index
        
class EMNISTDataset(Data):
    def __init__(self, X, y, green_idxs, red_idxs):
        super().__init__(X, y, green_idxs, red_idxs)
        self.transform = constants.EMNIST_TRANSFORM

    def __getitem__(self, index):
        img, target = self.X[index], self.y[index]

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index


class CustomDataset(Data):
    def __init__(self, X, y, green_idxs, red_idxs):
        super().__init__(X, y, green_idxs, red_idxs)
        self.transform = None

    def __getitem__(self, index):
        img, target = self.X[index], self.y[index]
        return img, target, index