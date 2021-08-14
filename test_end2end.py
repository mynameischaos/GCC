"""
Authors: Huasong Zhong
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os
import torch
import numpy as np

from utils.config import create_config
from utils.common_config import get_criterion, get_model, get_train_dataset,\
                                get_val_dataset, get_train_dataloader,\
                                get_val_dataloader, get_train_transformations,\
                                get_val_transformations, get_optimizer,\
                                adjust_learning_rate
from utils.evaluate_utils import contrastive_evaluate, get_predictions, hungarian_evaluate
from utils.memory import MemoryBank
from utils.train_utils import gcc_train
from utils.utils import fill_memory_bank, fill_memory_bank_mean
from termcolor import colored
from utils.aug_feat import AugFeat
from data import ConcatDataset

# Parser
parser = argparse.ArgumentParser(description='Graph Contrastive Clustering')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
args = parser.parse_args()

def main():
    org_feat_memory = AugFeat('./org_feat_memory', 4)
    aug_feat_memory = AugFeat('./aug_feat_memory', 4)

    # Retrieve config file
    print (args.config_env)
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))

    # Model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    print('Model is {}'.format(model.__class__.__name__))
    print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    #print(model)
    model = model.cuda()

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue'))
    torch.backends.cudnn.benchmark = True

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    train_transforms = get_train_transformations(p)
    print('Train transforms:', train_transforms)
    val_transforms = get_val_transformations(p)
    print('Validation transforms:', val_transforms)

    # Memory Bank
    print(colored('Build MemoryBank', 'blue'))
    base_dataset = get_train_dataset(p, val_transforms, to_end2end_dataset=True, split='train') # Dataset for performance test
    # for compare with SCAN
    #base_dataset = get_val_dataset(p, val_transforms, to_end2end_dataset=True) # Dataset for performance test
    base_dataloader = get_val_dataloader(p, base_dataset)
    print('Dataset contains {} test samples'.format(len(base_dataset)))

    memory_bank_base = MemoryBank(len(base_dataset),
                                p['model_kwargs']['features_dim'],
                                p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_base.cuda()

    # Checkpoint
    # end2end_model for kmeans model
    if os.path.exists(p['end2end_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['end2end_checkpoint']), 'blue'))
        checkpoint = torch.load(p['end2end_checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])

        # imagenet10
        #model_dict = model.state_dict()
        #pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        #model_dict.update(pretrained_dict)
        #model.load_state_dict(model_dict)

        model.cuda()
    else:
        print(colored('No checkpoint file at {}'.format(p['end2end_checkpoint']), 'blue'))
        exit(-1)

    fill_memory_bank(base_dataloader, model, memory_bank_base)
    #for topk in range(5, 51, 5):
    for topk in range(5, 6, 5):
        indices, acc, detail_acc = memory_bank_base.mine_nearest_neighbors(topk)
        print('Accuracy of top-%d nearest neighbors on val set is %.2f' %(topk, 100*acc))

    memory_bank_base.cpu()
    with open (p['features'], 'wb') as f:
        np.save(f, memory_bank_base.features)
    with open (p['features'] + "_label", 'wb') as f:
        np.save(f, memory_bank_base.targets)

    #from tsne import kmeans
    #kmeans(memory_bank_base.features.cpu().numpy(), memory_bank_base.targets.cpu().numpy())

    predictions, features, targets = get_predictions(p, base_dataloader, model, return_features=True)
    lowest_loss_head = 0
    clustering_stats = hungarian_evaluate(lowest_loss_head, predictions, compute_confusion_matrix=False)
    print(clustering_stats)

if __name__ == '__main__':
    main()
