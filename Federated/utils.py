import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import matplotlib
import matplotlib.font_manager
font = {'family' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)
import random
from copy import deepcopy
import json
import torch.nn as nn
import torch

def green_red_plot(dve_out, green_idxs, red_idxs, title, save_path=None):
    plt.clf()
    dve_out = np.squeeze(dve_out)
    plt.bar(green_idxs, dve_out[green_idxs], color="green")
    plt.bar(red_idxs, dve_out[red_idxs], color="red")
    plt.title(title)
    if save_path != None:
        plt.savefig(save_path)
    # plt.show()

def set_seed(seed:int=42):
    """Sets the seed for torch, numpy and random
    Args:
        seed (int): [description]
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_cuda_device(gpu_num: int):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

def dict_print(d:dict):
    d_new = deepcopy(d)

    def cast_str(d_new:dict):
        for k, v in d_new.items():
            if isinstance(v, dict):
                d_new[k] = cast_str(v)
            d_new[k] = str(v)
        return d_new
    d_new = cast_str(d_new)

    pretty_str = json.dumps(d_new, sort_keys=False, indent=4)
    print(pretty_str)
    return pretty_str

def init_weights(m:nn.Module):
    set_seed()
    def set_params(w):
        if isinstance(w, nn.Linear):
            torch.nn.init.xavier_uniform(w.weight)
            w.bias.data.fill_(0.01)
    m.apply(set_params)

def plot_fed_pred_accs(fed, pred, save_name):
    plt.clf()
    assert len(fed) == len(pred), "Pass consistentr values"
    x = np.arange(len(fed))
    msize = 8
    plt.plot(x, fed, scaley=False, color="black", 
            marker=".", markersize=msize, label="Federated Random Sampling",
            linestyle="-")
    plt.plot(x, pred, scaley=False, color="red", 
            marker=".", markersize=msize, label="Gradmatch",
            linestyle="-")

    plt.xlabel("Communication Round")
    plt.ylabel("Test Accuracy")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.5, linewidth=1, color="gray", linestyle=":")
    plt.title("Federated vs. Gradmatch")
    plt.savefig(f"Federated/logs/plots/{save_name}.png", dpi=300, bbox_inches = "tight")