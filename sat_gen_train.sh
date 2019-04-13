#!/bin/bash

python train.py hparams/GenMM/satimage1.json
python mat_acc.py results/satimage/k12hc8c1GenMM/satimage.json


python train.py hparams/GenMM/satimage2.json
python mat_acc.py results/satimage/k12hc8c2GenMM/satimage.json

python train.py hparams/GenMM/satimage4.json
python mat_acc.py results/satimage/k12hc8c4GenMM/satimage.json
