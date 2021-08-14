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
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))
    with open (p['log_output_file'], 'a+') as fw:
        fw.write(str(p) + "\n")

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
    train_dataset = get_train_dataset(p, train_transforms, to_end2end_dataset=True,
                                        split='train+unlabeled') # Split is for stl-10
    train_dataloader = get_train_dataloader(p, train_dataset)

    val_dataset = get_train_dataset(p, val_transforms, to_end2end_dataset=True,
                                        split='train') # Dataset w/o augs for knn eval
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(val_dataset)))

    # Memory Bank
    print(colored('Build MemoryBank', 'blue'))
    base_dataset = get_train_dataset(p, val_transforms, to_end2end_dataset=True, split='train') # Dataset for performance test
    base_dataloader = get_val_dataloader(p, base_dataset)
    memory_bank_base = MemoryBank(len(base_dataset),
                                p['model_kwargs']['features_dim'],
                                p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_base.cuda()
    memory_bank_val = MemoryBank(len(val_dataset),
                                p['model_kwargs']['features_dim'],
                                p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_val.cuda()

    # Criterion
    print(colored('Retrieve criterion', 'blue'))
    criterion1, criterion2 = get_criterion(p)
    print('Criterion is {}'.format(criterion1.__class__.__name__))
    print('Criterion is {}'.format(criterion2.__class__.__name__))
    criterion1 = criterion1.cuda()
    criterion2 = criterion2.cuda()

    # Optimizer and scheduler
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)

    # Checkpoint
    if os.path.exists(p['end2end_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['end2end_checkpoint']), 'blue'))
        checkpoint = torch.load(p['end2end_checkpoint'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        model.cuda()
        start_epoch = checkpoint['epoch'] + 10000 # 10000 for evaluate directly
    else:
        print(colored('No checkpoint file at {}'.format(p['end2end_checkpoint']), 'blue'))
        start_epoch = 0
        model = model.cuda()

    best_acc = 0.0
    # Training
    print(colored('Starting main loop', 'blue'))
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch, p['epochs']), 'yellow'))
        print(colored('-'*15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train
        if epoch <= 50:
            print('Train pretext...')
            gcc_train(train_dataloader, model, criterion1, criterion2, optimizer,
                    epoch, aug_feat_memory, org_feat_memory, p['log_output_file'], True)
        else:
            print('Train pretext and clustering...')
            gcc_train(train_dataloader, model, criterion1, criterion2, optimizer,
                    epoch, aug_feat_memory, org_feat_memory, p['log_output_file'], False)

        # Evaluate
        if epoch > 0 and epoch % 5 == 0:
            print ("Start to evaluate...")
            predictions = get_predictions(p, base_dataloader, model)
            lowest_loss_head = 0
            clustering_stats = hungarian_evaluate(lowest_loss_head, predictions, compute_confusion_matrix=False)
            print(clustering_stats, len(base_dataloader.dataset))
            with open (p['log_output_file'], 'a+') as fw:
                fw.write(str(clustering_stats) + "\n")

            if clustering_stats['ACC'] > best_acc:
                best_acc = clustering_stats['ACC']
                print ('Best acc: ', best_acc)
                # Checkpoint
                print('Checkpoint ...')
                torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                        'epoch': epoch + 1}, p['end2end_checkpoint'])

        # Update memory bank
        if epoch >= 50 and epoch % 5 == 0:
            if epoch == 50:
                train_dataset = get_train_dataset(p, train_transforms, to_end2end_dataset=True,
                        split='train') # Split is for stl-10
                train_dataloader = get_train_dataloader(p, train_dataset)

            # Fill memory bank
            topk = 5
            fill_memory_bank_mean(val_dataloader, aug_feat_memory, org_feat_memory,  memory_bank_val)
            indices, acc, detail_acc = memory_bank_val.mine_nearest_neighbors(topk)
            distance_dict = memory_bank_val.laplace_transform(indices)
            print('Accuracy of top-%d nearest neighbors on val set is %.2f' %(topk, 100*acc))
            with open (p['log_output_file'], 'a+') as fw:
                for acc in detail_acc:
                    fw.write(acc)
            np.save(p['topk_neighbors_val_path'], indices)
            train_dataset.update_neighbors(indices)
            train_dataset.update_distance(distance_dict)

    predictions, features, targets = get_predictions(p, base_dataloader, model, return_features=True)
    lowest_loss_head = 0
    clustering_stats = hungarian_evaluate(lowest_loss_head, predictions, compute_confusion_matrix=False)
    print(clustering_stats)

    with open (p['features'], 'wb') as f:
        np.save(f, features)
    with open (p['features'] + "_label", 'wb') as f:
        np.save(f, targets)

    # Save final model
    torch.save(model.state_dict(), p['end2end_model'])

if __name__ == '__main__':
    main()
