#!/bin/bash




python train.py hparams/LatMM/mnist1.json

python verify_acc.py results/mnist/k6hc128s1LatMM/mnist.json

