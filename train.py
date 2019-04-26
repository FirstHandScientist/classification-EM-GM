"""Train script.

Usage:
    train.py <hparams>
"""
import os
import torchvision.datasets as dset
import torch
import torchvision.utils as vutils

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
    hparams_dir = args["<hparams>"]
    assert os.path.exists(hparams_dir), (
        "Failed to find hparams josn `{}`".format(hparams))
    # dataset = args["<dataset>"]
    hparams = JsonConfig(hparams_dir)
    dataset = hparams.Data.dataset
    dataset_root = hparams.Data.dataset_root

    
    label_list = range(10)
    def worker(label):
        # load the subset data of the label
        local_hparams = JsonConfig(hparams_dir)

        local_hparams.Dir.log_root = os.path.join(local_hparams.Dir.log_root, "classfier{}".format(label))
        # warm_start = "results/cifar/k6hc64LatMM/classfier{}/log/save_0k100.pkg".format(label)
        # warm_start = "results/satimage/k8hc8c3GenMM/classfier{}/log/save_0k200.pkg".format(label)

        # if os.path.exists(warm_start):
        #     local_hparams.Train.warm_start = warm_start
        
        # if os.path.exists( os.path.join(local_hparams.Dir.log_root, "log/trained.pkg")):
        #     local_hparams.Train.warm_start = os.path.join(local_hparams.Dir.log_root, "log/trained.pkg")
        dataset = load_obj(os.path.join(dataset_root,  "classSets/" +"subset{}".format(label)))
        if True:
            tmp_dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                         shuffle=True, num_workers=int(2))
            img = next(iter(tmp_dataloader))
            
            if not os.path.exists(local_hparams.Dir.log_root):
                os.makedirs(local_hparams.Dir.log_root)

            vutils.save_image(img.data.add(0.5), os.path.join(local_hparams.Dir.log_root, "img_under_evaluation.png"))
      

        # dump the json file for performance evaluation
        if not os.path.exists(os.path.join(local_hparams.Dir.log_root, local_hparams.Data.dataset+ ".json")):
            get_hparams = JsonConfig(hparams_dir)
            data_dir = get_hparams.Data.dataset_root
            get_hparams.Data.dataset_root = data_dir.replace("separate", "all")
            get_hparams.dump(dir_path=get_hparams.Dir.log_root,
                             json_name=get_hparams.Data.dataset + ".json")

        ### build model and train
        built = build(local_hparams, True)
        
        print(hparams.Dir.log_root)
        trainer = Trainer(**built, dataset=dataset, hparams=local_hparams)
        trainer.train()

    # Parallel(n_jobs=10, pre_dispatch="all", backend="threading")(map(delayed(worker), [0]))
    Parallel(n_jobs=2, pre_dispatch="all", backend="threading")(map(delayed(worker), list(range(hparams.Data.num_classes))))
    
    # for the_label in label_list:
        
