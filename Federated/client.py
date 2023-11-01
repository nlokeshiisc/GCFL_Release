from time import time
from data import Data
from torch.utils.data import DataLoader
import copy
from cords.gradmatchstrategy import GradMatchStrategy
from cords.craigstrategy import CRAIGStrategy
from Crust.Strategy import CRUSTStrategy
import torch
import torch.nn as nn
import data_utils
import constants
from server import Server

class Client:
    
    def __init__(self, client_id, X, y, 
                 dataset_name,
                 green_idxs, red_idxs,
                 server_pred_model,
                 sampling_strategy="gm",
                 server_valid_ds = None,
                 sampling_pc=None,
                 num_labels=None,
                 gm_valgrad_reg = True,
                 subset_sel_rnd = 10,
                 warm_start=0,
                 logger=None):
        
        self.client_id = client_id
        self.dataset_name = dataset_name
        self.dataset_cls = data_utils.get_dataset_class(self.dataset_name)
        self.sampling_strategy = sampling_strategy

        #Create a object of Class Data and instantiate it and keep the full data 
        self.train_ds:Data = self.dataset_cls(X, y, green_idxs, red_idxs)
        self.train_loader = DataLoader(self.train_ds, batch_size=32, shuffle=False)

        # All of this is required only for gradmatch strategy
        if self.sampling_strategy in ["gm","craig","fl"]:
            self.gm_valgrad_reg = gm_valgrad_reg
            self.server_model = server_pred_model
            self.num_labels = num_labels

            if self.gm_valgrad_reg == True:
                '''We need server_valid_ds and server_valid_dataloader only when we do validation gradient regression'''
                self.server_valid_ds = server_valid_ds
                self.server_valid_loader = DataLoader(self.server_valid_ds, batch_size=32, shuffle=False)
            
            self.init_methods() 
            
            
        self.subset_sel_rnd = subset_sel_rnd

        self.sampling_percent = sampling_pc
        assert self.sampling_percent <= 1, "Pass a percentage for sampling budget"
        self.sampling_budget = int(sampling_pc * len(X))
        
        # If the client has too few data, it is best to exclude it completely from the Federated system altogether
        self.exclude_client = False
        if self.sampling_budget == 0:
            self.exclude_client = True
            logger.info(f"Client {self.client_id} has only budget = {self.sampling_budget} and thus excluded completely!!!")
        
        if not self.exclude_client:
            self.num_samples = len(X)
            self.warm_start = warm_start

            # These are kind of state variables that we use to compute the subset simply when we encounter epochs thta donot warrant subset selection
            if self.sampling_strategy == "fl":
                ds, _ = self.fl.select(self.sampling_budget,None)
                self.subset_idxs = torch.LongTensor(ds)
            else:
                self.subset_idxs = torch.randperm(self.num_samples)[:self.sampling_budget]
            self.subset_wts = torch.ones_like(self.subset_idxs) * (1/self.sampling_budget)
        
    def init_methods(self):
        """Initializes strategies. One time book keeping stuff.
        """
        if self.gm_valgrad_reg:
            target_loader = self.server_valid_loader
        else:
            target_loader = self.train_loader

        if self.sampling_strategy == "gm":
            self.gm  = GradMatchStrategy(trainloader=self.train_loader, 
                                            valloader=target_loader, 
                                            model=copy.deepcopy(self.server_model.to(constants.CUDA)),  
                                            loss=nn.CrossEntropyLoss(reduction='none'),
                                            eta=1e-2, device=constants.CUDA,
                                            num_classes=self.num_labels, 
                                            linear_layer=True,
                                            selection_type='PerClassPerGradient',
                                            valid=self.gm_valgrad_reg, 
                                            lam=0.5,
                                            verbose=False)

        elif self.sampling_strategy == "craig":
            self.craig  = CRUSTStrategy(trainloader=self.train_loader, 
                model=copy.deepcopy(self.server_model.to(constants.CUDA)),num_classes=self.num_labels,\
                loss=nn.CrossEntropyLoss(),device=constants.CUDA)

        elif self.sampling_strategy == "fl":
            self.fl = CRAIGStrategy(trainloader=self.train_loader, valloader=target_loader, \
                model=copy.deepcopy(self.server_model.to(constants.CUDA)),loss=nn.CrossEntropyLoss(reduction='none'),\
                    device=constants.CUDA,num_classes=self.num_labels, linear_layer=True,if_convex = True,\
                        selection_type='PerClass')

    def get_round_data(self, round_num, server_state_dict, logger=None):
        """Gets the dataset from client. 

        Args:
            round_num (_type_): Current Communication round for which dataset subset is being asked
            server_state_dict (_type_): model parameters of the server
            server_epochs (_type_): number of epochs to be run at the server
            logger (_type_, optional): _description_. Defaults to None.

        Returns:
            round_data
        """
        
        # Exclude the clients with too les data
        if self.exclude_client:
            return None, None

        def global_wts():
            wts = torch.zeros(self.num_samples)
            wts[self.subset_idxs] = self.subset_wts
        
        if (round_num+1) % self.subset_sel_rnd != 0:
            # Here we dont need subset selection. Thus we pass the subset that was selected earlier
            if self.sampling_strategy == "craig":
                self.subset_idxs, self.subset_wts = self.craig.select(self.sampling_percent, copy.deepcopy(server_state_dict))
                ds, _ = self.train_ds.index_sampler(self.subset_idxs)
            else:
                ds, _ = self.train_ds.index_sampler(self.subset_idxs)
        else:
            if self.sampling_strategy == "random":
                ds, self.subset_idxs = self.train_ds.random_sampler(self.sampling_budget)
            elif self.sampling_strategy == "gm":
                self.subset_idxs, self.subset_wts = self.gm.select(self.sampling_budget, copy.deepcopy(server_state_dict))
                ds, _ = self.train_ds.index_sampler(self.subset_idxs)
            elif self.sampling_strategy == "full":
                self.subset_wts = torch.ones(self.num_samples)/self.num_samples
                ds, self.subset_idxs = self.train_ds.all_sampler()
            elif self.sampling_strategy in ["random-warm","fl"]:
                ds, _ = self.train_ds.index_sampler(self.subset_idxs)
            elif self.sampling_strategy == "skyline":
                self.subset_wts = torch.ones(len(self.train_ds.green_idxs))/len(self.train_ds.green_idxs)
                ds, self.subset_idxs = self.train_ds.green_sampler_full()
            elif self.sampling_strategy == "craig":
                self.subset_idxs, self.subset_wts = self.craig.select(self.sampling_percent, copy.deepcopy(server_state_dict))
                ds, _ = self.train_ds.index_sampler(self.subset_idxs)
            else:
                raise ValueError()
        
        return ds, global_wts()