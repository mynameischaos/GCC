from PIL import Image
import os
import os.path
import numpy as np
import h5py
from skimage import io, color

import torchvision.datasets as datasets

from torch.utils.data.dataset import Dataset

class TinyImageNet(Dataset):
    """`TineyImageNet <https://cs.stanford.edu/~acoates/stl10/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``stl10_binary`` exists.
        split (string): One of {'train', 'test', 'unlabeled', 'train+unlabeled'}.
            Accordingly dataset is selected.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'tiny-imagenet-200'
    class_names_file = 'class_names.txt'
    train_list = [
        ['TinyImageNet.h5', '918c2871b30a85fa023e0c44e0bee87f'],
    ]

    splits = ('train', 'test')

    def __init__(self, split='train',
                 transform=None, target_transform=None, download=False):
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))

        root = '/gruntdata/dataset'
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # train/test/unlabeled set

        # now load the picked numpy arrays
        if self.split == 'train':
            self.data, self.labels = self.__loadfile(
                self.train_list[0][0])

        class_file = os.path.join(
            self.root, self.base_folder, self.class_names_file)
        if os.path.isfile(class_file):
            with open(class_file) as f:
                self.classes = f.read().splitlines()

        # consistent with other dataset
        self.targets = self.labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(np.transpose(np.uint8(img), (1, 2, 0)))
        img = Image.fromarray(np.uint8(img)).convert("RGB") # C,H,W -> H,W,C

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        out = {'image': img, 'target': target, 'meta': {'im_size': img.size, 'index': index, 'class_name': 'unlabeled'}}

        return out


    def __len__(self):
        return len(self.data)

    def __loadfile(self, data_file, labels_file=None):
        datas = []
        labels = []
        data_path = os.path.join(self.root, self.base_folder, 'tiny_imagenet_data_list.txt')
        with open (data_path, 'r') as fr:
            for line in fr.readlines():
                line = os.path.join(self.root, self.base_folder, line.strip())
                img = io.imread(line)
                img = color.gray2rgb(img)
                datas.append(img)
        label_path = os.path.join(self.root, self.base_folder, 'tiny_imagenet_label_list.txt')
        with open (label_path, 'r') as fr:
            for line in fr.readlines():
                labels.append(line.strip())

        return datas, labels

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)
