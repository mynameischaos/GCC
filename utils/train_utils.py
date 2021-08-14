"""
Authors: Huasong Zhong
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import numpy as np
from utils.utils import AverageMeter, ProgressMeter

def gcc_train(train_loader, model, criterion1, criterion2, optimizer, epoch, aug_feat_memory, org_feat_memory, log_output_file, only_train_pretext=True):
    """
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    losses = AverageMeter('Loss', ':.4e')
    constrastive_losses = AverageMeter('Constrast Loss', ':.4e')
    cluster_losses = AverageMeter('Cluster Loss', ':.4e')
    consistency_losses = AverageMeter('Consist Loss', ':.4e')
    entropy_losses = AverageMeter('Entropy Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [losses, constrastive_losses, cluster_losses, consistency_losses, entropy_losses],
        prefix="Epoch: [{}]".format(epoch), output_file=log_output_file)

    model.train()

    for i, batch in enumerate(train_loader):
        neighbor_top1_features = None
        neighbor_top2_features = None
        neighbor_top3_features = None
        if only_train_pretext:
            images = batch['image'].cuda(non_blocking=True)
            images_augmented = batch['augmented'].cuda(non_blocking=True)
        else:
            images = batch['image'].cuda(non_blocking=True)
            images_augmented = batch['augmented'].cuda(non_blocking=True)
            neighbor_top1 = batch['neighbor_top1'].cuda(non_blocking=True)
            neighbor_top2 = batch['neighbor_top2'].cuda(non_blocking=True)
            neighbor_top3 = batch['neighbor_top3'].cuda(non_blocking=True)
            neighbor_top1_features, neighbor_top1_cluster_outs = model(neighbor_top1)
            neighbor_top2_features, neighbor_top2_cluster_outs = model(neighbor_top2)
            neighbor_top3_features, neighbor_top3_cluster_outs = model(neighbor_top3)
            neighbor_top1_features = neighbor_top1_features * batch['neighbor_top1_weight'].unsqueeze(-1).cuda()
            neighbor_top2_features = neighbor_top2_features * batch['neighbor_top2_weight'].unsqueeze(-1).cuda()
            neighbor_top3_features = neighbor_top3_features * batch['neighbor_top3_weight'].unsqueeze(-1).cuda()
            b = batch['neighbor_top1_weight'].shape[0]
            fill_one_diag_zero = torch.ones([b, b]).fill_diagonal_(0).cuda()
            neighbor_weights = torch.cat([fill_one_diag_zero + torch.diag(batch['neighbor_top1_weight'].cuda()),
                                          fill_one_diag_zero + torch.diag(batch['neighbor_top2_weight'].cuda()),
                                          fill_one_diag_zero + torch.diag(batch['neighbor_top3_weight'].cuda())], dim=1)

        neighbors = batch['neighbor'].cuda(non_blocking=True)

        b, c, h, w = images.size()
        input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
        input_ = input_.view(-1, c, h, w)
        input_ = input_.cuda(non_blocking=True)
        targets = batch['target'].cuda(non_blocking=True)
        constrastive_features, cluster_outs = model(input_)
        constrastive_features = constrastive_features.view(b, 2, -1)

        if not only_train_pretext:
            neighbor_topk_features = torch.cat([neighbor_top1_features, neighbor_top2_features, neighbor_top3_features], dim=0).cuda()
            constrastive_loss = criterion1(constrastive_features, neighbor_topk_features, neighbor_weights, 3)
        else:
            constrastive_loss = criterion1(constrastive_features, None, None, 0)

        aug_feat_memory.push(constrastive_features.clone().detach()[:, 1], batch['meta']['index'])

        if not only_train_pretext:
            neighbors_features, neighbors_output = model(neighbors)

            # Loss for every head
            total_loss, consistency_loss, entropy_loss = [], [], []
            for image_and_aug_output_subhead, neighbor_top1_cluster, neighbor_top2_cluster, neighbor_top3_cluster in \
                    zip(cluster_outs, neighbor_top1_cluster_outs, neighbor_top2_cluster_outs, neighbor_top3_cluster_outs):
                image_and_aug_output_subhead = image_and_aug_output_subhead.view(b, 2, -1)
                image_output_subhead = image_and_aug_output_subhead[:, 0]
                #aug_output_subhead = image_and_aug_output_subhead[:, 1]
                neightbor_output_subhead = neighbors_output[0]
                total_loss_, consistency_loss_, entropy_loss_ = criterion2(image_output_subhead,
                                                                           neightbor_output_subhead,
                                                                           #aug_output_subhead,
                                                                           )
                total_loss.append(total_loss_)
                consistency_loss.append(consistency_loss_)
                entropy_loss.append(entropy_loss_)
            cluster_loss = torch.sum(torch.stack(total_loss, dim=0))

            #cluster_loss = torch.tensor([0.0]).cuda()
        else:
            cluster_loss = torch.tensor([0.0]).cuda()

        loss = 2.0 * constrastive_loss + cluster_loss

        losses.update(loss.item())
        constrastive_losses.update(constrastive_loss.item())
        cluster_losses.update(cluster_loss.item())
        if not only_train_pretext:
            consistency_losses.update(np.mean([v.item() for v in consistency_loss]))
            entropy_losses.update(np.mean([v.item() for v in entropy_loss]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)


def selflabel_train(train_loader, model, criterion, optimizer, epoch, ema=None, output_file=None):
    """
    Self-labeling based on confident samples
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), [losses],
                                prefix="Epoch: [{}]".format(epoch), output_file=output_file)
    model.train()

    for i, batch in enumerate(train_loader):
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)

        with torch.no_grad():
            output = model(images)[0]
        output_augmented = model(images_augmented)[0]

        loss = criterion(output, output_augmented)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema is not None: # Apply EMA to update the weights of the network
            ema.update_params(model)
            ema.apply_shadow(model)

        if i % 25 == 0:
            progress.display(i)
