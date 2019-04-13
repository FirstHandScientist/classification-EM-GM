#!/bin/bash

python train.py hparams/LatMM/letter1.json 
python mat_acc.py results/letter/k12hc8s1LatMM/letter.json

python train.py hparams/LatMM/letter2.json 
python mat_acc.py results/letter/k12hc8s2LatMM/letter.json


python train.py hparams/LatMM/letter3.json 
python mat_acc.py results/letter/k12hc8s3LatMM/letter.json
