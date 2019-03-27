"""Train script.

Usage:
    train.py <hparams>
"""
import os
import torchvision.datasets as dset
from docopt import docopt
from torchvision import transforms
from glow.builder import build
from glow.trainer import Trainer
from glow.config import JsonConfig
from backup_glow.utils import load_obj
from joblib import Parallel, delayed
from torch.utils.data import DataLoader


if __name__ == "__main__":
    args = docopt(__doc__)
    hparams = args["<hparams>"]
    # dataset = args["<dataset>"]
    dataset = "cifar10"
    dataset_root = "/home/doli/datasets/cifar10"

    assert os.path.exists(hparams), (
        "Failed to find hparams josn `{}`".format(hparams))

    # dataset_root = args["<dataset_root>"]
    # assert os.path.exists(dataset_root), (
    #     "Failed to find root dir `{}` of dataset.".format(dataset_root))
    hparams = JsonConfig(hparams)


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
        
    label_list = range(10)
    def worker(label):
        # load the subset data of the label
        local_hparams = JsonConfig(hparams)
        local_hparams.Dir.log_root = os.path.join("results", "FullLatMMcifarH6Gamma", "classfier{}".format(label))
        # if os.path.exists( os.path.join(local_hparams.Dir.log_root, "log/trained.pkg")):
        #     local_hparams.Train.warm_start = os.path.join(local_hparams.Dir.log_root, "log/trained.pkg")
        dataset = load_obj(dataset_root + "/classSets/" +"subset{}".format(label))
            
        built = build(local_hparams, True)
        
        print(hparams.Dir.log_root)
        trainer = Trainer(**built, dataset=dataset, hparams=local_hparams)
        trainer.train()

    # Parallel(n_jobs=10, pre_dispatch="all", backend="threading")(map(delayed(worker), [0,1,2,3,4,5,6,7,8,9]))
    Parallel(n_jobs=10, pre_dispatch="all", backend="threading")(map(delayed(worker), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

    # for the_label in label_list:
        
