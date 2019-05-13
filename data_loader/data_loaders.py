from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.hand_dataset_ox import Ox_hand
from data_loader.hand_seg_RHD import Handseg_RHD
from data_loader.Huawei_dataset import Huawei_Data
from data_loader.Group_transform import *



class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super(MnistDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class OxHandDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True, vis=False):
        trsfm = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
    ])
        self.data_dir = data_dir
        self.dataset = Ox_hand(self.data_dir, train=training, download=True, transform=trsfm, vis=vis)
        super(OxHandDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
        
        
class HandSegRHD(BaseDataLoader):
    
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True, vis=False):
        trsfm = transforms.Compose([
            transforms.RandomCrop((256,256)),
            # transforms.Resize((256, 256)),
            transforms.ColorJitter(brightness=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        self.data_dir = data_dir
        self.dataset = Handseg_RHD(self.data_dir, train=training, download=True, transform=trsfm, vis=vis)
        super(HandSegRHD, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class Huawei(BaseDataLoader):
    
    def __init__(self, data_dir, list_path,  batch_size, shuffle, validation_split, num_workers, training=True, vis=False):
        trsfm = transforms.Compose([
            GroupRandomCrop((256,256)),
            # transforms.Resize((256, 256)),
            ToTensor(),
            GroupNormalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225]),
        ])
        self.data_dir = data_dir
        self.list_path = list_path
        self.dataset = Huawei_Data(self.data_dir, list_path=self.list_path, is_training=training, transform=trsfm)
        super(Huawei, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, )
