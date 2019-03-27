import os
import torchvision.datasets as dset
from torch.utils.data import DataLoader
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


def training(hparams):

    dataset = hparams.Data.dataset
    dataset_root = hparams.Dir.dataset_root
    if dataset == "mnist":
        transform = transforms.Compose([
            transforms.Resize(hparams.Data.resize),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        dataset = dset.MNIST(root=dataset_root,
                         download=True,
                         transform=transform)
    elif dataset == "fashion-mnist":
        transform = transforms.Compose([
            transforms.Resize(hparams.Data.resize),
            transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        dataset = dset.FashionMNIST(root=dataset_root,
                         download=True,
                         transform=transform)
    elif dataset == "cifar10":
        #dataset = load_obj(datasat_root + "/classSets/" +"subset0")
        pass
        
    label_list = range(hparams.Data.num_classes)

    for the_label in label_list:
        # load the subset data of the label
        subset_dir = os.path.join(dataset_root, "classSets", "subset{}".format(the_label))
        dataset = load_obj(subset_dir)
        built = build(hparams, True)
        trainer = Trainer(**built, dataset=dataset, hparams=hparams, label=the_label)
        trainer.train()
    # def training_job(the_label):
    #     # load the subset data of the label
    #     subset_dir = os.path.join(dataset_root, "classSets", "subset{}".format(the_label))
    #     dataset = load_obj(subset_dir)
    #     built = build(hparams, True)
    #     trainer = Trainer(**built, dataset=dataset, hparams=hparams, label=the_label)
    #     trainer.train()
    
    # _ = Parallel(n_jobs=3)(delayed(training_job)(i) for i in label_list)

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
    dataset_root = hparams.Dir.dataset_root
    #hparams = CLASSIFIER_DIR.format(the_label)
    # hparams = JsonConfig("results/H256cifar10/LatMMK5/class1/cifar10.json")
    # hparams.Infer.pre_trained = "tet"
    ####  set data loader
    batch_size = hparams.Train.batch_size
    transform = transforms.Compose([
            transforms.ToTensor()
        ])
    # test accuracy
    # step_batches = np.linspace(hparams.Train.checkpoints_gap,
    #                           hparams.Train.num_batches,
    #                           hparams.Train.num_batches//hparams.Train.em_gap)
    # step_batches = np.linspace(78*6,
    #                           hparams.Train.num_batches,
    #                           78*hparams.Train.em_gap)
    # step_batches = np.arange(start=132, stop=132*12, step=132)
    step_batches = [4290]
    accuracy_dict = {"train":[], "test":[]}
    # accuracy_dict = {"test":[]}

    for step in step_batches:
        if step<0:
            continue
        # net_name = "save_{}k{}.pkg".format(int(step)//1000, int(step)%1000)
        net_name = "trained.pkg"

        for key, value in accuracy_dict.items():
            dataset = dset.CIFAR10(root=dataset_root, train=True if key=="train" else False,
                           download=True,
                           transform=transform)
            data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             #   num_workers=8,
                             shuffle=True,
                             drop_last=True)
            accuracy = 0
            myclassifer = load_classifier(net_name)
            progress = tqdm(data_loader)
            for i_batch, batch in enumerate(progress):

                x_testing = batch[0].to(hparams.Device.data) * 256
                y_testing = batch[1].to(hparams.Device.data)

                prediction = []
                for the_classifier in myclassifer:
                    # _, gaussian_nlogp, nlogdet, _ = the_classifier(x=x_testing, reverse=False)
                    # prior = the_classifier.get_prior()
                    # nll = (gaussian_nlogp + nlogdet) * ( x_testing.size(1)*x_testing.size(2)*x_testing.size(3)) * prior.unsqueeze(1).expand_as(gaussian_nlogp).to(hparams.Device.data)
                    nll = nll_compute(graph=the_classifier,
                                      naive=hparams.Mixture.naive,
                                      x=x_testing)
                    
                    #nll = nll.sum(axis=0)
                    
                    prediction.append(nll)

                #prediction = torch.stack(prediction)
                prediction = np.array(prediction)

                y_predition = prediction.argmin(axis=0)
                accuracy += (y_testing.cpu().data.numpy()==y_predition).sum()
            # report the accuracy
            #accuracy = accuracy.type(torch.DoubleTensor)/(batch_size*(i_batch+1))
            accuracy = accuracy/(batch_size*(i_batch+1))

            print("[Step {}, State {}, Accuracy: {}]".format(step, key, accuracy))
            #value.append(accuracy.cpu().data.numpy())
            value.append(accuracy)

    accuracy_dict["step"] = step_batches
    return accuracy_dict
    
if __name__ == "__main__":
    WorkDir = "results/FullLatMMcifarH6"
    hparams =JsonConfig(os.path.join(WorkDir,"cifar10.json"))

    #hparams.Dir.classifier_dir = os.path.join(hparams.Dir.log_root,"class{}/checkpoints/")
    hparams.Dir.log_root = WorkDir
    hparams.Dir.classifier_dir = os.path.join(hparams.Dir.log_root,"classfier{}/log")

    hparams.Data.dataset = "cifar10"
    hparams.Dir.dataset_root = "/home/doli/datasets/cifar10"
    hparams.Data.num_classes = 10
    log_dir = hparams.Dir.log_root
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    hparams.dump(log_dir, json_name="cifar10.json")
    #testing(hparams)
    
    accuracy_dict = testing(hparams)
    score_dir = os.path.join(hparams.Dir.log_root, "accuracy.pkl")
    with open(score_dir, "wb") as f:
        pickle.dump(accuracy_dict, f)
    
    value_pd = pd.DataFrame(accuracy_dict)
    print(value_pd)
    with open(os.path.join(hparams.Dir.log_root, "accuracy.tex"), 'w') as tf:
        tf.write(value_pd.to_latex())
