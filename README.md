Code for reproducing experiment (classification part) results of [our paper](https://arxiv.org/abs/1907.13432)
===============================================================

## Requirements:
    * Python 3.6.5
    * Pytorch 0.4.1
    * torchvision 0.2.1
    * tqdm 4.26.0
    * numpy, scipy

## Datasets:
Algorithm evaluation in our experiment section is done with dataset:
   
* Satimage
* Letter
* Norb
    

## Usage Instruction:

### Train model with:


$ python train.py <hparams> 

 
If hparams.Mixture.naive = True in configuration file <hparams>, then GenMM is going to be trained. If hparams.naive is setting False, LatMM is going to be trained.
    
Training example:

$ python train.py hparams/LatMM/fashion-mnist5.json


### Classification performance evaluation:

$ python mat_acc.py <hparams>

