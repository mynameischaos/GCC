# Graph Contrastive Clustering

This repo contains the Pytorch implementation of our paper:
> [Graph Contrastive Clustering](https://arxiv.org/abs/2104.01429)
>
> Huasong Zhong, Jianlong Wu, Chong Chen and so on.

## Contents

1. [Introduction](#introduction)
0. [Installation](#installation)
0. [Training](#train)
0. [Testing](#test)
0. [Self-labeling](#self-labeling)
0. [Citation](#citation)

## Introduction
<p align="center">
    <img src="images/pre.jpg" />

<p align="center">
    <img src="images/main.png" />

## Installation
```shell
pip install -r requirements.txt
```

## Train
CUDA\_VISIBLE\_DEVICES=0 python end2end.py --config\_env configs/env.yml --config\_exp configs/end2end/end2end\_cifar10.yml
CUDA\_VISIBLE\_DEVICES=0 python end2end.py --config\_env configs/env.yml --config\_exp configs/end2end/end2end\_cifar20.yml
CUDA\_VISIBLE\_DEVICES=0 python end2end.py --config\_env configs/env.yml --config\_exp configs/end2end/end2end\_imagenet10.yml
CUDA\_VISIBLE\_DEVICES=0 python end2end.py --config\_env configs/env.yml --config\_exp configs/end2end/end2end\_imagenet\_dogs.yml
CUDA\_VISIBLE\_DEVICES=0 python end2end.py --config\_env configs/env.yml --config\_exp configs/end2end/end2end\_tiny\_imagenet.yml
CUDA\_VISIBLE\_DEVICES=0 python end2end.py --config\_env configs/env.yml --config\_exp configs/end2end/end2end\_stl10.yml

## Test
CUDA\_VISIBLE\_DEVICES=0 python test\_end2end.py --config\_env configs/env.yml --config\_exp configs/end2end/end2end\_cifar10.yml
CUDA\_VISIBLE\_DEVICES=0 python test\_end2end.py --config\_env configs/env.yml --config\_exp configs/end2end/end2end\_cifar20.yml
CUDA\_VISIBLE\_DEVICES=0 python test\_end2end.py --config\_env configs/env.yml --config\_exp configs/end2end/end2end\_imagenet10.yml
CUDA\_VISIBLE\_DEVICES=0 python test\_end2end.py --config\_env configs/env.yml --config\_exp configs/end2end/end2end\_imagenet\_dogs.yml
CUDA\_VISIBLE\_DEVICES=0 python test\_end2end.py --config\_env configs/env.yml --config\_exp configs/end2end/end2end\_tiny\_imagenet.yml
CUDA\_VISIBLE\_DEVICES=0 python test\_end2end.py --config\_env configs/env.yml --config\_exp configs/end2end/end2end\_stl10.yml


## Self-labeling
CUDA\_VISIBLE\_DEVICES=0 python selflabel.py --config\_env configs/env.yml --config\_exp configs/selflabel/selflabel\_cifar10.yml
CUDA\_VISIBLE\_DEVICES=0 python selflabel.py --config\_env configs/env.yml --config\_exp configs/selflabel/selflabel\_cifar20.yml
CUDA\_VISIBLE\_DEVICES=0 python selflabel.py --config\_env configs/env.yml --config\_exp configs/selflabel/selflabel\_stl10.yml

## Citation 

If you use GCC in your research or wish to refer to the baseline results published in this paper, please use the following BibTeX entry.

@article{zhong2021graph,
  title={Graph Contrastive Clustering},
  author={Zhong, Huasong and Wu, Jianlong and Chen, Chong and Huang, Jianqiang and Deng, Minghua and Nie, Liqiang and Lin, Zhouchen and Hua, Xian-Sheng},
  journal={arXiv preprint arXiv:2104.01429},
  year={2021}
}
