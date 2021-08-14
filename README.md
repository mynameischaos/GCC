# Graph Contrastive Clustering

This repo contains the Pytorch implementation of our paper:
> [Graph Contrastive Clustering](https://arxiv.org/abs/2104.01429)
>
> Huasong Zhong, Jianlong Wu, Chong Chen, Jianqiang Huang, 
> Minghua Deng, Liqiang Nie, Zhouchen Lin, Xian-Sheng Hua
- __Accepted at ICCV 2021.


## Contents

1. [Introduction](#introduction)
0. [Installation](#installation)
0. [Training](#train)
0. [Testing](#test)
0. [Self-labeling](#self-labeling)
0. [Citation](#citation)

## Introduction

- Motivation of GCC. (a) Existing contrastive
learning based clustering methods mainly focus on instancelevel consistency, which maximizes the correlation between selfaugmented samples and treats all other samples as negative samples. (b) GCC incorporates the category information to perform
the contrastive learning at both the instance and the cluster levels,
which can better minimize the intra-cluster variance and maximize
the inter-cluster variance.

<p align="center" width="128" height="256">
    <img src="images/pre.jpg" />

-  Framework of the proposed Graph Contrastive Clustering. GCC has two heads with shared CNN parameters. The first head is a
representation graph contrastive (RGC) module, which helps to learn clustering-friendly features. The second head is an assignment graph
contrastive (AGC) module, which leads to a more compact cluster assignment.

<p align="center">
    <img src="images/main.jpg" />

## Installation
```shell
pip install -r requirements.txt
```

## Train
```shell
CUDA\_VISIBLE\_DEVICES=0 python end2end.py --config\_env configs/env.yml --config\_exp configs/end2end/end2end\_cifar10.yml
CUDA\_VISIBLE\_DEVICES=0 python end2end.py --config\_env configs/env.yml --config\_exp configs/end2end/end2end\_cifar20.yml
CUDA\_VISIBLE\_DEVICES=0 python end2end.py --config\_env configs/env.yml --config\_exp configs/end2end/end2end\_imagenet10.yml
CUDA\_VISIBLE\_DEVICES=0 python end2end.py --config\_env configs/env.yml --config\_exp configs/end2end/end2end\_imagenet\_dogs.yml
CUDA\_VISIBLE\_DEVICES=0 python end2end.py --config\_env configs/env.yml --config\_exp configs/end2end/end2end\_tiny\_imagenet.yml
CUDA\_VISIBLE\_DEVICES=0 python end2end.py --config\_env configs/env.yml --config\_exp configs/end2end/end2end\_stl10.yml
```

## Test
```shell
CUDA\_VISIBLE\_DEVICES=0 python test\_end2end.py --config\_env configs/env.yml --config\_exp configs/end2end/end2end\_cifar10.yml
CUDA\_VISIBLE\_DEVICES=0 python test\_end2end.py --config\_env configs/env.yml --config\_exp configs/end2end/end2end\_cifar20.yml
CUDA\_VISIBLE\_DEVICES=0 python test\_end2end.py --config\_env configs/env.yml --config\_exp configs/end2end/end2end\_imagenet10.yml
CUDA\_VISIBLE\_DEVICES=0 python test\_end2end.py --config\_env configs/env.yml --config\_exp configs/end2end/end2end\_imagenet\_dogs.yml
CUDA\_VISIBLE\_DEVICES=0 python test\_end2end.py --config\_env configs/env.yml --config\_exp configs/end2end/end2end\_tiny\_imagenet.yml
CUDA\_VISIBLE\_DEVICES=0 python test\_end2end.py --config\_env configs/env.yml --config\_exp configs/end2end/end2end\_stl10.yml
```

## Self-labeling
```shell
CUDA\_VISIBLE\_DEVICES=0 python selflabel.py --config\_env configs/env.yml --config\_exp configs/selflabel/selflabel\_cifar10.yml
CUDA\_VISIBLE\_DEVICES=0 python selflabel.py --config\_env configs/env.yml --config\_exp configs/selflabel/selflabel\_cifar20.yml
CUDA\_VISIBLE\_DEVICES=0 python selflabel.py --config\_env configs/env.yml --config\_exp configs/selflabel/selflabel\_stl10.yml
```

## Citation 

If you use GCC in your research or wish to refer to the baseline results published in this paper, please use the following BibTeX entry.

```bibtex
@article{zhong2021graph,
  title={Graph Contrastive Clustering},
  author={Zhong, Huasong and Wu, Jianlong and Chen, Chong and Huang, Jianqiang and Deng, Minghua and Nie, Liqiang and Lin, Zhouchen and Hua, Xian-Sheng},
  journal={arXiv preprint arXiv:2104.01429},
  year={2021}
}
```
