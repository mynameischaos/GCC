"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch
from torch.utils.data import Dataset

"""
    AugmentedDataset
    Returns an image together with an augmentation.
"""
class AugmentedDataset(Dataset):
    def __init__(self, dataset):
        super(AugmentedDataset, self).__init__()
        transform = dataset.transform
        dataset.transform = None
        self.dataset = dataset

        if isinstance(transform, dict):
            self.image_transform = transform['standard']
            self.augmentation_transform = transform['augment']

        else:
            self.image_transform = transform
            self.augmentation_transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset.__getitem__(index)
        image = sample['image']

        sample['image'] = self.image_transform(image)

        sample['image_augmented'] = self.augmentation_transform(image)

        return sample


"""
    NeighborsDataset
    Returns an image with one of its neighbors.
"""
class NeighborsDataset(Dataset):
    def __init__(self, dataset, indices, num_neighbors=None):
        super(NeighborsDataset, self).__init__()
        transform = dataset.transform

        if isinstance(transform, dict):
            self.anchor_transform = transform['standard']
            self.neighbor_transform = transform['augment']
        else:
            self.anchor_transform = transform
            self.neighbor_transform = transform

        dataset.transform = None
        self.dataset = dataset
        self.indices = indices # Nearest neighbor indices (np.array  [len(dataset) x k])
        if num_neighbors is not None:
            self.indices = self.indices[:, :num_neighbors+1]
        assert(self.indices.shape[0] == len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)

        neighbor_index = np.random.choice(self.indices[index], 1)[0]
        neighbor = self.dataset.__getitem__(neighbor_index)

        anchor['image'] = self.anchor_transform(anchor['image'])
        neighbor['image'] = self.neighbor_transform(neighbor['image'])

        output['anchor'] = anchor['image']
        output['neighbor'] = neighbor['image']
        output['possible_neighbors'] = torch.from_numpy(self.indices[index])
        output['target'] = anchor['target']

        return output

"""
    End2End
    Returns an image with one of its neighbors and their augmentation
"""
class End2EndDataset(Dataset):
    def __init__(self, dataset, indices, num_neighbors=None):
        super(End2EndDataset, self).__init__()
        transform = dataset.transform

        if isinstance(transform, dict):
            self.anchor_transform = transform['standard']
            self.neighbor_transform = transform['augment']
            self.augmentation_transform = transform['augment']
        else:
            self.anchor_transform = transform
            self.neighbor_transform = transform
            self.augmentation_transform = transform

        dataset.transform = None
        self.dataset = dataset
        #self.targets = dataset.targets
        self.indices = indices # Nearest neighbor indices (np.array  [len(dataset) x k])

        #if num_neighbors is not None:
        #    self.indices = self.indices[:, :num_neighbors+1]

        self.distance = None

    def __len__(self):
        return len(self.dataset)

    def update_neighbors(self, indices):
        print ("update neighbors!!!")
        self.indices = indices

    def update_distance(self, distance):
        print ("update distance!!!")
        self.distance = distance

    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)
        image = anchor['image']
        output['augmented'] = self.augmentation_transform(image)
        output['meta'] = anchor['meta']

        if self.indices is None or self.distance is None:
            neighbor_index = index
        else:
            neighbor_index = np.random.choice(self.indices[index], 1)[0]
            output['neighbor_top1'] = self.neighbor_transform(self.dataset.__getitem__(self.indices[index][0])['image'])
            output['neighbor_top2'] = self.neighbor_transform(self.dataset.__getitem__(self.indices[index][1])['image'])
            output['neighbor_top3'] = self.neighbor_transform(self.dataset.__getitem__(self.indices[index][2])['image'])
            output['neighbor_top1_weight'] = self.distance[self.indices[index][0]]
            output['neighbor_top2_weight'] = self.distance[self.indices[index][1]]
            output['neighbor_top3_weight'] = self.distance[self.indices[index][2]]

        neighbor = self.dataset.__getitem__(neighbor_index)

        anchor['image'] = self.anchor_transform(image)
        neighbor['image'] = self.neighbor_transform(neighbor['image'])
        output['image'] = anchor['image']
        output['neighbor'] = neighbor['image']

        if self.indices is not None:
            output['possible_neighbors'] = torch.from_numpy(self.indices[index])
        else:
            output['possible_neighbors'] = torch.from_numpy(np.array([index]))

        output['target'] = anchor['target']

        return output
