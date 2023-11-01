import numpy as np
import constants
from torch.utils.data import DataLoader
from data import CustomDataset
import data_utils
import torch
from utils import init_weights
from utils import set_seed
import models
import logging
from collections import Counter
from fed_avg import Fed_avg

class Server:
    
    def __init__(self, X, y,
                 dataset_name,
                 num_features = None, 
                 num_classes = None,
                 model_type=constants.NN,
                 model_arch=None,
                 batch_size = 16,
                 global_lr=1e-2,
                 client_ids=None,
                 split_per={"train":0, "test":0.5, "valid":0.5},
                 num_rounds = None,
                 fed_epochs=None,
                 val_size=None,
                 logger:logging.Logger = None):

        self.split_per = split_per
        self.client_ids = client_ids
        self.model_type = model_type
        self.dataset_name = dataset_name

        self.dataset_cls = data_utils.get_dataset_class(self.dataset_name)

        self.model_arch = model_arch
        self.batch_size = batch_size        
        self.learning_rate = global_lr
        self.num_rounds = num_rounds
        self.fed_epochs = fed_epochs
        self.val_size = val_size
        self.num_classes = num_classes # Server should have data of all labels
        self.num_features = num_features

        if num_classes is None:
            self.num_classes = len(torch.unique(y))
        if num_features is None and self.model_type == constants.NN :
            #assert self.model_type == constants.NN "If u try other models, be sure to pass the num_features appropriately" 
            self.num_features = X.shape[1]
        
        #Create a object of Class Data and instantiate it and keep the full data 
        self.full_data = self.dataset_cls(X, y, None, None)
        self.distribute_data(logger)

        self.model = None
        self.setup_classifier()

        self.setup_fedavg()
       
    def distribute_data(self, logger:logging.Logger):
        #Create a object of Class Data and instantiate it and keep the full data as train data
        X = self.full_data.X
        y = self.full_data.y
        per = [self.split_per["train"], self.split_per["test"], self.split_per["valid"]]
        
        _, test, val = data_utils.split_data(X, y, per)

        if self.val_size is not None:
            # This is for the ablation study on the validation dataset size
            assert self.val_size <= 1., "pass a valid percentage"

            _, _, val = data_utils.split_data(val[0], val[1], per = [0, 1-self.val_size, self.val_size])

            print(f"New valid data has {len(val[0])} samples")
            if logger is not None:
                logger.info(f"New valid data has {len(val[0])} samples")
        
        # self.train = self.dataset_cls(train[0], train[1], None, None) # Server need not possess a trainloader
        self.test_ds = self.dataset_cls(test[0], test[1], None, None)
        self.valid_ds:CustomDataset = self.dataset_cls(val[0], val[1], None, None)

        # logger.info(f"Server test label dist: {Counter(self.test_ds[1])}")
        # logger.info(f"Server valid label dist: {Counter(self.valid_ds[1])}")

        # self.train_loader = DataLoader(self.train, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)
        self.val_loader = DataLoader(self.valid_ds, batch_size=len(self.valid_ds), shuffle=False) # We set the length as 1 to be able to compute the  mean val gradient easily

    def setup_classifier(self):
        """Creates the classifier model.
        """
        kwargs = {
            constants.HID_DIM: self.model_arch
        }
        self.model = models.get_pred_model(model_type=self.model_type,
                                            dataset_name=self.dataset_name,
                                            num_classes=self.num_classes,
                                            num_features=self.num_features,
                                            **kwargs)
        # We set seed here so that models start with the same initialization
        set_seed()
        init_weights(self.model)

    def setup_fedavg(self):
        """Sets up the Federated model. It accepts the classifier and implements the code to perform federated avg
        and stuff.
        """
        self.fed_model = Fed_avg(model=self.model, global_lr=self.learning_rate)

    def get_cls_state(self):
        """Returnd the state dict of the classifier model

        Returns:
            _type_: _description_
        """
        return self.model.state_dict()
        
    def get_round_clients(self, client_sel_strategy="all", round_clients=2):
        """Server samples the clients in each roiund according to the client sampling strategy
        """
        if client_sel_strategy == "all":
            return self.client_ids
        if client_sel_strategy == "random":
            """
            This does random sampling of clients
            """
            return np.random.choice(self.client_ids, round_clients, replace=False)
        else:
            assert False, "Client selection strategy {} is not handled".format(client_sel_strategy)

    def update_fed_model(self, clients_data, round_num, fed_epochs=1, fed_batch=32):
        """Updates the Federated model with the client's data.

        Args:
            clients_data (_type_): _description_
            round_num (_type_): _description_

        Returns:
            _type_: The performance of the federated model after the updates based on the client's data dicts.
        """
        clients_fed_data = {}
        client_weights = {}
        """Let us not use the federated weights for the time being"""
        for client_id, data_dict in  clients_data.items():
            clients_fed_data[client_id] = data_dict[constants.FED_CLIDATA]
            client_weights[client_id] = None#data_dict[constants.FED_CLIWTS]

        # Update the model with fed averaging on best batch
        self.fed_model.federated_avg(clients_ds_dict=clients_fed_data, clients_weights=client_weights, 
                                        epoch=fed_epochs, batch=fed_batch, verbose=False)
        return self.fed_model.evaluate_model(self.test_loader)