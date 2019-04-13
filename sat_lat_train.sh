#!/bin/bash

python train.py hparams/LatMM/satimage1.json
python mat_acc.py results/satimage/k12hc8s1LatMM/satimage.json


python train.py hparams/LatMM/satimage2.json
python mat_acc.py results/satimage/k12hc8s2LatMM/satimage.json

python train.py hparams/LatMM/satimage3.json
python mat_acc.py results/satimage/k12hc8s3LatMM/satimage.json

python train.py hparams/LatMM/satimage4.json
python mat_acc.py results/satimage/k12hc8s4LatMM/satimage.json
