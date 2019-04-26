#!/bin/bash

# python train.py hparams/GenMM/cifar101.json

# python verify_acc.py results/cifar/k12hc64c1GenMM/cifar10.json

python verify_acc.py results/cifar/k6hc128c1GenMM/cifar10.json &
python verify_acc.py results/cifar/k6hc256c1GenMM/cifar10.json &
