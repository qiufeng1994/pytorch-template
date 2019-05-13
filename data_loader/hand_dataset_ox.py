from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
import random
from path import Path
from scipy.misc import imread
import scipy.io as sio
class Ox_hand(data.Dataset):
    """`Ox <http://www.robots.ox.ac.uk/~vgg/data/hands/index.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    raw_folder = 'raw'
    training_file = 'training_dataset/training_data'
    test_file = 'test_dataset/test_data'
    valid_file = 'validation_dataset/validation_data'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, vis=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.vis = vis
        if self.train:
            self.train_root = (Path(self.root) / self.training_file / 'images')
            self.train_samples = self.collect_samples(self.train_root, self.training_file)
            
            self.validation_root = (Path(self.root) / self.valid_file / 'images')
            self.validation_samples = self.collect_samples(self.validation_root, self.valid_file)
        else:
            self.test_root = (Path(self.root) / self.test_file / 'images')
            self.test_samples = self.collect_samples(self.test_root, self.test_file)
    
    def collect_samples(self, root, file):
        samples = []
        for img in sorted((root).glob('*.jpg')):
            _img = img.basename().split('.')[0]
            label = (Path(self.root) / file / 'annotations' / _img + '.mat')
            assert label.exists()
            try:
                mat = sio.loadmat(label)
                box_num = len(mat['boxes'][0])
                for i in range(box_num):
                    box = np.array([e for e in mat['boxes'][0][i].ravel()[0]])
                    box = box
                sample = {'img': img, 'label': label}
                samples.append(sample)
            except:
                print('data {} error'.format(img))
        return samples
    
    # def load_samples(self):
    
    def load_samples(self, s):
        image = imread(s['img'], mode='RGB')
        mat = sio.loadmat(s['label'])
        box_num = len(mat['boxes'][0])
        boxes = []
        for i in range(box_num):
            box = np.array([e for e in mat['boxes'][0][i].ravel()[0]])
            boxes.extend([np.array([e for e in np.array(box[:4])]).reshape(8)])
        return [image, boxes[random.randrange(0,box_num)]]
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        
        if self.train:
            s = self.train_samples[index]
        else:
            s = self.test_samples[index]
        image, target = self.load_samples(s)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.array(image), mode='RGB')
        h, w = img.size[0], img.size[1]
        target[::2] = target[::2] / w
        target[1::2] = target[1::2] / h
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.vis == True:
            return img, target, image
        
        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_samples)
        else:
            return len(self.test_samples)


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

def visual_box(data, output, i):
    import cv2
    import scipy
    img = Image.fromarray(np.array(data).squeeze(), mode='RGB')
    h, w = img.size[0], img.size[1]
    output = np.array(output).squeeze()
    output[::2] = output[::2] * w
    output[1::2] = output[1::2] * h
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    for j in range(0,8,2):
        cv2.circle(img, (int(output[j+1]), int(output[j])), 5, (255,255,0), -1)
    # box format (w1, h1, w2, h2, ...)
    cv2.imwrite('/Data/hand_dataset_ox/vis/{:05d}.jpg'.format(i), img)
    print('img saving to \'/Data/hand_dataset_ox/vis/{:05d}.jpg\''.format(i))
    
if __name__ == '__main__':
    visual_box()