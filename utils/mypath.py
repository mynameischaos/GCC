"""
Author: Huasong Zhong
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'cifar-10', 'stl-10', 'cifar-20', 'imagenet', 'imagenet_50', 'imagenet_100', 'imagenet_200', 'tiny_imagenet'}
        assert(database in db_names)

        if database == 'cifar-10':
            return '/gruntdata/dataset'

        elif database == 'cifar-20':
            return '/gruntdata/dataset'

        elif database == 'stl-10':
            return '/gruntdata/dataset'

        elif database == 'tiny_imagenet':
            return '/gruntdata/dataset'

        elif database in ['imagenet', 'imagenet_50', 'imagenet_100', 'imagenet_200']:
            return '/path/to/imagenet/'

        else:
            raise NotImplementedError
