"""
Author: Huasong Zhong
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import json
import torch
import pickle
import numpy as np

class AugFeat:
    def __init__(self, output_path, size=20, alpha=0.3):
        self.memory_dict = {}
        self.output_path = output_path
        self.size = size
        self.alpha = alpha

    @torch.no_grad()
    def push(self, feats, indexes):
        for i in range(indexes.shape[0]):
            key = indexes[i].item()
            #data = feats[i][0].cpu().detach().numpy()
            data = feats[i].unsqueeze_(0)
            if key not in self.memory_dict.keys():
                self.memory_dict[key] = [data]
                #self.memory_dict[key] = data
            elif len(self.memory_dict[key]) == self.size:
                self.memory_dict[key].pop(0)
                self.memory_dict[key].append(data)
            else:
                self.memory_dict[key].append(data)
                #self.memory_dict[key] = self.alpha * self.memory_dict[key] + (1.0 - self.alpha) * data

    @torch.no_grad()
    def pop(self, key):
        return self.memory_dict[key]

    def save(self):
        np.save(self.output_path, self.memory_dict)
