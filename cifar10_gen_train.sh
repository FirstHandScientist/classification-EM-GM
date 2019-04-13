#!/bin/bash

python train.py hparams/GenMM/cifar101.json

python verify_acc.py results/cifar/k12hc64c1GenMM/cifar10.json

