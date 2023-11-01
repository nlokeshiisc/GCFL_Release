import numpy as np
import copy
from sklearn.model_selection import train_test_split
import torch.optim as optim
from data import CIFARDataset, Data, EMNISTDataset, CustomDataset, SVHNDataset, FLOWERSDataset
import constants
import torch
import random
import time
from imagecorruptions import corrupt, get_corruption_names

def split_data(X, y, per):
    
    assert type(X) == torch.Tensor, "please pass a valid Tensor"

    assert sum(per) == 1, "pass a valid split criterion"
    assert len(per) == 3, "currently we only handle train test valid split"
    assert  len(X) == len(y), "pass corresponding X and y"
    
    per_list = []
    list_len = len(X)
    for entry in per[:-1]:
        per_list.append(int(entry * list_len))

    per_list.append(list_len - sum(per_list))
    
    X_tr, X_te, X_va = torch.split(X, per_list)
    y_tr, y_te, y_va = torch.split(y, per_list)
    
    return (X_tr, y_tr), (X_te, y_te), (X_va, y_va)

def flip_labels(y_train, noise_rate, y_set=None):
    """Flips the y_train labels randomly for the noise_rate proprortion of them

    Args:
        y_train (_type_): _description_
        noise_rate (_type_): _description_
        y_set (_type_, optional): _description_. Defaults to None.

    Returns:
        corrupted labels and the corrupted indices
    """
    if y_set is None:
        y_set = list(set(np.squeeze(y_train)))
    
    rng = np.random.default_rng(12345)
    
    temp_idx = rng.permutation(len(y_train))
    noise_idx = temp_idx[:int(len(y_train) * noise_rate)]
    corrupted_y_train = copy.deepcopy(y_train[:])
    for itt in noise_idx:
        temp_y_set = copy.deepcopy(y_set)
        del temp_y_set[y_train[itt]]
        rand_idx = rng.permutation(len(y_set) - 1)[0]
        corrupted_y_train[itt] = temp_y_set[rand_idx]
    
    return (torch.LongTensor(corrupted_y_train), torch.LongTensor(noise_idx))

def flip_openlabels(y_train, green_y, red_y):
    """Flips all the red labels randomly to one of the green labels.

    Args:
        y_train (_type_): _description_
        noise_rate (_type_): _description_
        y_set (_type_, optional): _description_. Defaults to None.

    Returns:
        corrupted labels and the corrupted indices
    """
    red_idxs = []
    y = []
    for idx, yy in enumerate(y_train):
        if yy in green_y:
            y.append(yy)
        else:
            # If the index is a red idx, flip it randomly to one of the green labels and record it as a noisy index
            y.append(np.random.choice(green_y))
            red_idxs.append(idx)
    return (torch.LongTensor(y), torch.LongTensor(red_idxs))

def add_atttr_noise(x_train, noise_pc):
    """Corrupts the attributes with noise available from the imagecorruption library. We add 9 kinds of noises

    Args:
        x_train (_type_): _description_
        noise_pc (_type_): _description_

    Returns:
        Corrupted features, idxs of noisy attributes
    """
    corruption_types = get_corruption_names()
    corruption_types = [corruption_types[entry] for entry in [0, 1, 2, 8, 9, 10, 12, 13, 14]]

    rng = np.random.default_rng(12345)
    temp_idx = rng.permutation(len(x_train))
    noise_idx = temp_idx[:int(len(x_train) * noise_pc)]    

    def do_corruption(x):
        x = x.numpy().astype(np.uint8)
        corrupted_image = corrupt(x, corruption_name=np.random.choice(corruption_types), severity=np.random.randint(5, 6))
        return torch.FloatTensor(corrupted_image)
    
    for idx in noise_idx:
        x_train[idx] = do_corruption(x_train[idx])
    
    return x_train, noise_idx
    

def train_test_valid_split(X, y, split_per={"train":0.60, "test": 0.2, "val":0.2}):
    test_val_per = split_per["test"]+split_per["val"]
    X_train, X_test, y_train, y_test  = train_test_split(X, y, stratify=y, 
                                                         test_size=test_val_per)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, stratify=y_test,
                                                    test_size=split_per["val"]/test_val_per)
    return X_train, y_train, X_test, y_test, X_val, y_val


def get_lr_scheduler(optimizer, scheduler_name, n_rounds=None):
    """
    Gets torch.optim.lr_scheduler given an lr_scheduler name and an optimizer
    :param optimizer:
    :type optimizer: torch.optim.Optimizer
    :param scheduler_name: possible are
    :type scheduler_name: str
    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`
    :type n_rounds: int
    :return: torch.optim.lr_scheduler
    """

    if scheduler_name == "sqrt":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / np.sqrt(x) if x > 0 else 1)

    elif scheduler_name == "linear":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / x if x > 0 else 1)

    elif scheduler_name == "constant":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)

    elif scheduler_name == "cosine_annealing":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)

    elif scheduler_name == "multi_step":
        assert n_rounds is not None, "Number of rounds is needed for \"multi_step\" scheduler!"
        milestones = [n_rounds//2, 3*(n_rounds//4)]
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    else:
        raise NotImplementedError("Other learning rate schedulers are not implemented")

def get_dataset_class(dataset_name) -> Data:
    if dataset_name == constants.EMNIST:
        return EMNISTDataset
    elif dataset_name in [constants.CIFAR_10,constants.CIFAR_100]:
        return CIFARDataset
    if dataset_name == constants.FEMNIST:
        return EMNISTDataset
    elif dataset_name == constants.CIFAR_100:
        return CIFARDataset
    elif dataset_name == constants.SVHN:
        return SVHNDataset
    elif dataset_name == constants.FLOWERS:
        return FLOWERSDataset
    elif dataset_name == constants.FLOWERS_INCEPTION:
        return CustomDataset
    elif dataset_name == constants.CIFAR_10_RESNET50:
        return CustomDataset
    elif dataset_name == constants.CIFAR_10_MOBILENETV2:
        return CustomDataset
    elif dataset_name == constants.CIFAR_100_MOBILENETV2:
        return CustomDataset
    elif dataset_name == constants.ADULT:
        return CustomDataset
    elif dataset_name == constants.MUSHROOM:
        return CustomDataset
    elif dataset_name == constants.MONK:
        return CustomDataset
    elif dataset_name == constants.SEGMENT:
        return CustomDataset
    elif dataset_name in [constants.LETTER,constants.SENSORLESS,constants.SHUTTLE,constants.USPS]:
        return CustomDataset
    else:
        raise NotImplementedError()
    
    
def iid_divide(data_ids, num_clients):
    """
    https://github.com/TalwalkarLab/leaf/blob/master/data/utils/sample.py
    divide list `l` among `g` groups
    each group has either `int(len(l)/g)` or `int(len(l)/g)+1` elements
    returns a list of groups
    """
    num_elems = len(data_ids)
    group_size = int(len(data_ids) / num_clients)
    num_big_groups = num_elems - num_clients * group_size
    num_small_groups = num_clients - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(data_ids[group_size * i: group_size * (i + 1)])
    bi = group_size * num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(data_ids[bi + group_size * i:bi + group_size * (i + 1)])
    return glist


def split_list_by_indices(data_idxs, grp_idx_start):
    """
    divide list `l` given indices into `len(indices)` sub-lists
    sub-list `i` starts from `grp_idx_start[i]` and stops at `indices[i+1]`
    returns a list of sub-lists
    """
    res = []
    current_index = 0
    for index in grp_idx_start:
        res.append(data_idxs[current_index: index])
        current_index = index
    return res


def dirichlet_split(labels, n_clients, n_clusters=None, alpha=0.4, frac=1, seed=1234):
    """
    split classification dataset among `n_clients`. The dataset is split as follow:
        1) classes are grouped into `n_clusters`
        2) for each cluster `c`, samples are partitioned across clients using dirichlet distribution
    Inspired by the split in "Federated Learning with Matched Averaging"__(https://arxiv.org/abs/2002.06440)
    :param dataset:
    :type dataset: torch.utils.Dataset
    :param n_classes: number of classes present in `dataset`
    :param n_clients: number of clients
    :param n_clusters: number of clusters to consider; if it is `-1`, then `n_clusters = n_classes`
    :param alpha: parameter controlling the diversity among clients
    :param frac: fraction of dataset to use
    :param seed:
    :return: list (size `n_clients`) of subgroups, each subgroup is a list of indices.
    """
    n_classes = len(torch.unique(labels))
    
    if n_clusters is None:
        n_clusters = int(n_classes)

    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    all_labels = list(range(n_classes))
    rng.shuffle(all_labels)
    clusters_labels = iid_divide(all_labels, n_clusters)

    label2cluster = dict()  # maps label to its cluster
    for group_idx, grp_labels in enumerate(clusters_labels):
        for l in grp_labels:
            label2cluster[l] = group_idx

    # get subset
    n_samples = int(len(labels) * frac)
    selected_indices = rng.sample(list(range(len(labels))), n_samples)

    clusters_sizes = np.zeros(n_clusters, dtype=int)
    clusters = {k: [] for k in range(n_clusters)}
    for idx in selected_indices:
        label = labels[idx].item()
        group_id = label2cluster[label]
        clusters_sizes[group_id] += 1
        clusters[group_id].append(idx)

    for _, cluster in clusters.items():
        rng.shuffle(cluster)

    clients_counts = np.zeros((n_clusters, n_clients), dtype=np.int64)  # number of samples by client from each cluster

    for cluster_id in range(n_clusters):
        weights = np.random.dirichlet(alpha=alpha * np.ones(n_clients))
        clients_counts[cluster_id] = np.random.multinomial(clusters_sizes[cluster_id], weights)

    clients_counts = np.cumsum(clients_counts, axis=1)

    clients_indices = [[] for _ in range(n_clients)]
    for cluster_id in range(n_clusters):
        cluster_split = split_list_by_indices(clusters[cluster_id], clients_counts[cluster_id])

        for client_id, indices in enumerate(cluster_split):
            clients_indices[client_id] += indices

    clients_indices = [ torch.LongTensor(entry) for entry in clients_indices ]
    
    return clients_indices


def pathological_split(labels, n_clients, n_classes_per_client=1, frac=1, seed=1234):
    """
    split classification dataset among `n_clients`. The dataset is split as follow:
        1) sort the data by label
        2) divide it into `n_clients * n_classes_per_client` shards, of equal size.
        3) assign each of the `n_clients` with `n_classes_per_client` shards
    Inspired by the split in
     "Communication-Efficient Learning of Deep Networks from Decentralized Data"__(https://arxiv.org/abs/1602.05629)
    :param dataset:
    :type dataset: torch.utils.Dataset
    :param n_classes: umber of classes present in `dataset`
    :param n_clients: number of clients
    :param n_classes_per_client:
    :param frac: fraction of dataset to use
    :param seed:
    :return: list (size `n_clients`) of subgroups, each subgroup is a list of indices.
    """
    n_classes = len(torch.unique(labels))

    labels = labels.tolist()
    
    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    # get subset
    n_samples = int(len(labels) * frac)
    selected_indices = rng.sample(list(range(len(labels))), n_samples)

    label2index = {k: [] for k in range(n_classes)}
    for idx in selected_indices:
        label = labels[idx]
        label2index[label].append(idx)

    sorted_indices = []
    for label in label2index:
        sorted_indices += label2index[label]

    n_shards = n_clients * n_classes_per_client
    shards = iid_divide(sorted_indices, n_shards)
    random.shuffle(shards)
    tasks_shards = iid_divide(shards, n_clients)

    clients_indices = [[] for _ in range(n_clients)]
    for client_id in range(n_clients):
        for shard in tasks_shards[client_id]:
            clients_indices[client_id] += shard

    clients_indices = [torch.LongTensor(entry) for entry in clients_indices]
    
    return clients_indices



def setdiff1d(t1, t2):
    """Computes the Set difference t1 - t2
    Args:
        t1 (_type_): _description_
        t2 (_type_): _description_
    """
    if len(t1) == 0 or len(t2) == 0:
        return t1
    combined = torch.cat((t1, t2, t2))
    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]
    return difference