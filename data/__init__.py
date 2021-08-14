from torch.utils.data.dataset import ConcatDataset as _ConcatDataset_
from functools import reduce

class ConcatDataset(_ConcatDataset_):
    """Dataset as a concatenation of multiple datasets

    Wrapper class of Pytorch ConcatDataset to set the labels as an attribute

    """

    def __init__(self, *args, **kwargs):
        super(ConcatDataset, self).__init__(*args, **kwargs)
        self.targets = reduce(lambda x,y:x+y.targets, self.datasets, [])
