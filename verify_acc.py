"""Train script.
Usage:
    verify_cc.py <hparams>
"""

import os, glob
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch
from docopt import docopt
from torchvision import transforms
from glow.builder import build
from glow.trainer import Trainer
from glow.config import JsonConfig
from glow import thops
from backup_glow.utils import load_obj
from tqdm import tqdm
import numpy as np
import pickle
from joblib import Parallel, delayed
import pandas as pd

def load_nll(scalar_dir=None):
    file_dir = os.path.join(scalar_dir, "events.out.tfevents.*.gpu1")
    file_dir = glob.glob(file_dir)[-1]
    event_acc = EventAccumulator(file_dir)
    event_acc.Reload()
        # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
    w_times, step_nums, vals = zip(*event_acc.Scalars('nll_value/step'))
    vals = np.array(vals)
    
    return vals.min()

def nll_scale(dum_dir):

    nll_list = []
    label_list = range(hparams.Data.num_classes)
    for the_label in label_list:
        current_dir = dum_dir.format(the_label)
        nll_list.append(load_nll(current_dir))

    nll_array = np.array(nll_list)
    return nll_array

def load_classifier(net_name):
        myclassifer = []
        CLASSIFIER_DIR = os.path.join(hparams.Dir.classifier_dir, net_name)
        label_list = range(hparams.Data.num_classes)

        for the_label in label_list:
            print("[Loading classifer: {}]".format(the_label))
            hparams.Infer.pre_trained = CLASSIFIER_DIR.format(the_label)
            built = build(hparams, False)
            built["graph"].get_component().eval()
            myclassifer.append(built["graph"])
        return myclassifer

def nll_compute(graph, naive, x):
    y_onehot = None
    with torch.no_grad():
        if naive:
            z, nll = graph(x=x, y_onehot=y_onehot)
                    
            logp = -nll.cpu().numpy()
        else:
            z, gaussian_nlogp, nlogdet, reg_prior_logp = graph(x=x, y_onehot=y_onehot,regulate_std=False)
            #testing reverse
            logp = -(gaussian_nlogp + nlogdet) 
            logp = logp.cpu().numpy()
        logp = logp * thops.pixels(x)
        #min_logp = logp.min(axis= 0)
        min_logp = logp.mean(axis=0)
        delta_logp = logp - min_logp
        delta_logp = delta_logp.astype(np.float128)
        summand = np.exp(delta_logp) * graph.get_prior().numpy()[:,None].astype(np.float128)
        log_sum = np.log(np.sum( summand, axis=0) )
        loss = (-log_sum - min_logp)/thops.pixels(x)
    return loss

def testing(hparams):

    dataset = hparams.Data.dataset
    dataset_root = hparams.Data.dataset_root
    ####  set data loader
    batch_size = hparams.Train.batch_size

    step_batches = np.arange(start=0, stop=hparams.Train.n_epoches, step=hparams.Train.em_gap)

    accuracy_dict = {"test":[], "train":[]}

    for step in step_batches:
        if step<0:
            continue
        net_name = "save_{}k{}.pkg".format(int(step)//1000, int(step)%1000)
        # net_name = "trained.pkg"

        for key, value in accuracy_dict.items():
            if dataset == "cifar10":
                transform = transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (1,1,1))
                ])
                dataset_ins = dset.CIFAR10(root=dataset_root, train=True if "train" == key else False,
                               download=True,
                               transform=transform)
            elif dataset == "mnist":
                transform = transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (1,))
                ])
                dataset_ins = dset.MNIST(root=dataset_root, train=True if "train" == key else False,
                               download=False,
                               transform=transform)
            elif dataset == "fashion-mnist":
                transform = transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (1,))
                ])
                dataset_ins = dset.FashionMNIST(root=dataset_root, train=True if "train" == key else False,
                               download=False,
                               transform=transform)

            data_loader = DataLoader(dataset_ins,
                             batch_size=batch_size,
                             #   num_workers=8,
                             shuffle=True,
                             drop_last=True)
            accuracy = 0
            myclassifer = load_classifier(net_name)
            progress = tqdm(data_loader)
            count = 0
            for i_batch, batch in enumerate(progress):

                x_testing = batch[0].to(hparams.Device.data) 
                y_testing = batch[1].to(hparams.Device.data)

                prediction = []
                for idx, the_classifier in enumerate(myclassifer):

                    nll = nll_compute(graph=the_classifier,
                                      naive=hparams.Mixture.naive,
                                      x=x_testing)
                    prediction.append(nll)


                prediction = np.array(prediction)
                prediction[prediction==-np.inf] = np.inf
                y_predition = prediction.argmin(axis=0)
                accuracy += (y_testing.cpu().data.numpy()==y_predition).sum()
                count += 1
            # report the accuracy
            #accuracy = accuracy.type(torch.DoubleTensor)/(batch_size*(i_batch+1))
            accuracy = accuracy/(batch_size*(count))

            print("[Step {}, State {}, Accuracy: {}]".format(step, key, accuracy))
            #value.append(accuracy.cpu().data.numpy())
            value.append(accuracy)
            data_loader = None

    accuracy_dict["step"] = step_batches
    return accuracy_dict
    
if __name__ == "__main__":
    args = docopt(__doc__)
    hparams_dir = args["<hparams>"]
    assert os.path.exists(hparams_dir), (
        "Failed to find hparams josn `{}`".format(hparams))
    
    hparams =JsonConfig(hparams_dir)
    hparams.Dir.log_root = os.path.dirname(hparams_dir)
    hparams.Dir.classifier_dir = os.path.join(hparams.Dir.log_root,"classfier{}/log")

    log_dir = hparams.Dir.log_root
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    
    accuracy_dict = testing(hparams)

    score_dir = os.path.join(hparams.Dir.log_root, "accuracy.pkl")
    with open(score_dir, "wb") as f:
        pickle.dump(accuracy_dict, f)
    
    value_pd = pd.DataFrame(accuracy_dict)
    print(value_pd)
    with open(os.path.join(hparams.Dir.log_root, "accuracy.tex"), 'w') as tf:
        tf.write(value_pd.to_latex())
