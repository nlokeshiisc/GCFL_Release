from sklearn.metrics import pairwise_distances
from Crust.Crust import estimate_grads
from Crust.fl_cifar import FacilityLocationCIFAR
from Crust.lazyGreedy import lazy_greedy_heap
import numpy as np
import torch

class CRUSTStrategy(object):

    def __init__(self, trainloader, model, num_classes,  loss, device):
        """
        Constructer method
        """
        self.trainloader = trainloader  # assume its a sequential loader.
        self.model = model
        self.num_classes = num_classes
        self.loss = loss
        self.device = device

    def select(self,sampling_budget, state_dict):

        self.model.load_state_dict(state_dict)

        grads_all, labels = estimate_grads(self.trainloader, self.model, self.loss,self.device)
        # per-class clustering
        ssets = []
        weights = []
        for c in range(self.num_classes):
            sample_ids = np.where((labels == c) == True)[0]
            if len(sample_ids) == 0:
                continue
            grads = grads_all[sample_ids]
            
            dists = pairwise_distances(grads)
            weight = np.sum(dists < 2.0, axis=1)
            V = range(len(grads))
            Fac = FacilityLocationCIFAR(V, D=dists)
            B = int(sampling_budget * len(grads))
            if B == 0:
                continue
            sset, vals = lazy_greedy_heap(Fac, V, B)
            #print(sset,B)
            weights.extend(weight[sset].tolist())
            sset = sample_ids[np.array(sset)]
            ssets += list(sset)
            
        weights = torch.FloatTensor(weights)
        ssets = torch.LongTensor(ssets)
        return ssets, weights