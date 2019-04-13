#!/bin/bash




python train.py hparams/LatMM/norb1.json

python mat_acc.py results/norb/k6hc64s1LatMM/norb.json


# python train.py hparams/LatMM/norb2.json
# python mat_acc.py results/norb/k6hc64s2LatMM/norb.json

# python train.py hparams/LatMM/norb3.json
# python mat_acc.py results/norb/k6hc64s3LatMM/norb.json

# python train.py hparams/LatMM/norb4.json
# python mat_acc.py results/norb/k6hc64s4LatMM/norb.json
