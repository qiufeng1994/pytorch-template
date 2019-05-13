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
from scipy.misc import imread, imresize
import scipy.io as sio


class Handseg_RHD(data.Dataset):
    """`Ox <https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html>`_ Dataset.

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
    training_file = 'training/'
    test_file = 'evaluation/'
    
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, vis=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.vis = vis
        if self.train:
            self.train_root = (Path(self.root) / self.training_file / 'color')
            self.train_samples = self.collect_samples(self.train_root, self.training_file)
        else:
            self.test_root = (Path(self.root) / self.test_file / 'color')
            self.test_samples = self.collect_samples(self.test_root, self.test_file)
    
    def collect_samples(self, root, file):
        samples = []
        for img in sorted((root).glob('*.png')):
            _img = img.basename().split('.')[0]
            label = (Path(self.root) / file / 'hand_mask' / _img + '.png')
            if self.train:
                assert label.exists()
            sample = {'img': img, 'label': label}
            samples.append(sample)
        return samples
        
    def load_samples(self, s):
        image = imread(s['img'], mode='RGB')
        try:
            label = imread(s['label'], mode='L')
        except:
            label = image
        return [image, label]

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
        # target = Image.fromarray(np.array(image))
        h, w = img.size[0], img.size[1]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        target = imresize(target, (256, 256))
        
        hand_mask = (target / 255).astype('uint8')
        bg_mask = np.logical_not((target/255).astype('uint8')).astype('uint8')
        target = np.stack((bg_mask, hand_mask), axis=2)
        if self.vis:
            return img, target.astype('float'), image
        return img, target.astype('float')
    
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
    for j in range(0, 8, 2):
        cv2.circle(img, (int(output[j + 1]), int(output[j])), 5, (255, 255, 0), -1)
    # box format (w1, h1, w2, h2, ...)
    cv2.imwrite('/Data/hand_dataset_ox/vis/{:05d}.jpg'.format(i), img)
    print('img saving to \'/Data/hand_dataset_ox/vis/{:05d}.jpg\''.format(i))

def process_hand_mask():
    import cv2
    from collections import Counter
    root = Path('/Data/RHD_v1-1/RHD_published_v2')
    file = 'evaluation'
    masks = sorted((root / file / 'mask').glob('*.png'))
    hand_mask_dir = root / file / 'hand_mask'
    hand_mask_dir.mkdir_p()
    print('total mask png {}'.format(len(masks)))
    for i in range(len(masks)):
        print('processing {}/{}'.format(i, len(masks)))
        mask = cv2.imread(masks[i],cv2.IMREAD_GRAYSCALE)
        mask_hand = mask > 1 # True for hand, False for bg
        mask_hand = mask_hand.astype('uint8') * 255
        cv2.imwrite(hand_mask_dir / masks[i].basename(), mask_hand)
    print(1)
    
def visual_mask(image, output, target, i ):
    import cv2
    save_root = Path('./visual_mask')
    print('saving to {}/{:05d}'.format(save_root, i))
    target = torch.argmax(target, dim=3).squeeze() * 255

    output = torch.argmax(output.permute([0,2,3,1]), dim=3).squeeze() * 255
    # output =
    # cv2.imwrite(save_root / '{:05}_mask_hand.png'.format(i), (np.array(target.squeeze()[:,:,1])*255).astype('uint8'))
    # cv2.imwrite(save_root / '{:05}_mask_hand.png'.format(i), (np.array(target.squeeze()[:, :, 0])*255).astype('uint8'))
    output = np.array(output).astype('uint8')
    target = np.array(target).astype('uint8')
    image = np.array(image).squeeze().astype('uint8')
    cv2.imwrite(save_root / '{:05}.png'.format(i), cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    cv2.imwrite(save_root / '{:05}_pre.png'.format(i), output)
    cv2.imwrite(save_root / '{:05}_mask.png'.format(i), target)

if __name__ == '__main__':
    process_hand_mask()
    # import shutil
    #
    # data = Path('/Data/RHD_v1-1/RHD_published_v2/evaluation/')
    # imgs = data.glob("*.png")
    # imgs.sort()
    #
    # for i in range(len(imgs)):
    #     shutil.copyfile(imgs[i], data/'color'/"real_{:05d}.png".format(i))