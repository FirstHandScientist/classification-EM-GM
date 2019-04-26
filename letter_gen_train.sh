#!/bin/bash

# python train.py hparams/GenMM/letter1.json 
# python mat_acc.py results/letter/k12hc8c1GenMM/letter.json

# python train.py hparams/GenMM/letter2.json 
# python mat_acc.py results/letter/k12hc8c2GenMM/letter.json


# python train.py hparams/GenMM/letter4.json 
# python mat_acc.py results/letter/k12hc8c4GenMM/letter.json


python train_letter.py hparams/GenMM/letter10.json 
python mat_acc.py results/letter/k12hc8c10GenMM/letter.json

python train_letter.py hparams/GenMM/letter20.json 
python mat_acc.py results/letter/k12hc8c20GenMM/letter.json
