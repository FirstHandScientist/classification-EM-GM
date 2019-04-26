import pickle
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import numpy as np

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

components = [1, 2, 3, 4, 10, 20]
dataset = ["satimage", "letter", "norb"]
my_dict = {
    "satimage": "results/archive/satimage/k12hc8c{}GenMM/accuracy",
    "letter": "results/letter/k12hc8c{}GenMM/accuracy",
    "norb": "results/archive/norb/k6hc64c{}GenMM/accuracy"
}
SAVE_DIR = "pictures/figures"
for name in dataset:
    for c in components:
        record = load_obj(name=my_dict[name].format(c))
        
        fig, ax = plt.subplots()
        ax.plot("step", "train", data=record, linestyle="--", label="Train")
        ax.plot("step", "test", data=record, label="Test")
        ax.legend()        
        plt.grid(True)
        plt.tight_layout()
        
        plt.xlim(0,300)
        
        fig.savefig(fname=os.path.join(SAVE_DIR, name + "_{}.pdf".format(c)))
        
        
