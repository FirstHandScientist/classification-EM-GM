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

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


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
            #######################nats/pixels#################
            # real_p = np.exp(logp) * self.graph.get_prior().numpy()[:, np.newaxis]
            # tmp_sum = np.sum(real_p, axis=0)
            # loss = np.mean( - np.log(tmp_sum + 1e-6) )
            #######################exactly compute#################
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

    # step_batches = np.linspace(78*6,
    #                           hparams.Train.num_batches,
    #                           78*hparams.Train.em_gap)
    step_batches = np.arange(start=0, stop=hparams.Train.n_epoches, step=hparams.Train.em_gap)
    # step_batches = [4290]
    accuracy_dict = {"test":[], "train":[]}
    # accuracy_dict = {"test":[]}
    #scaling_nll = nll_scale(hparams.Dir.classifier_dir)
    for step in step_batches:
        if step<0:
            continue
        net_name = "save_{}k{}.pkg".format(int(step)//1000, int(step)%1000)
        # net_name = "trained.pkg"

        for key, value in accuracy_dict.items():
            #if dataset == "vowel":
            data_x_y = load_obj(os.path.join(dataset_root,  "train_data" if key == "train" else "test_data"))
            data_x = data_x_y[0]
            data_y = data_x_y[1]
                
            accuracy = 0
            myclassifer = load_classifier(net_name)
            progress = tqdm(range(int(data_x.shape[0]/batch_size)))
            count = 0
            for batch_i in progress:

                x_testing = data_x[batch_i*batch_size:(batch_i+1)*batch_size].to(hparams.Device.data) 
                y_testing = data_y[batch_i*batch_size:(batch_i+1)*batch_size].to(hparams.Device.data)

                prediction = []
                for idx, the_classifier in enumerate(myclassifer):

                    nll = nll_compute(graph=the_classifier,
                                      naive=hparams.Mixture.naive,
                                      x=x_testing)
                    prediction.append(nll)

                #prediction = torch.stack(prediction)
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
