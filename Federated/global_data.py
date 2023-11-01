"""
This file loads the dataset and handles partitioning of data across clients and server.
"""

import pickle as pkl
import data_utils as du
import constants
import torch
import logging

import torchvision
from torchvision.transforms import Resize


class Global_data:
    def __init__(self, dataset_name, dist_strategy, num_clients, 
                 noise_type, noise_pc, noise_variation, logger:logging.Logger, **kwargs):
        self.dataset_name = dataset_name
        self.dist_strategy = dist_strategy
        self.num_clients = num_clients
        self.noise_type = noise_type
        self.noise_pc = noise_pc
        self.noise_variation = noise_variation
        if self.noise_type == constants.OPENNOISE:
            assert "red_labels" in kwargs and "green_labels" in kwargs, "we need these for openset noise for sure"
            # assert self.dist_strategy == constants.IID, "For now I assume that open set noise follows iid split only. If we want to support non-iid also, we have to see how to split the openset labels in a non-iid manner across clients"
            
            self.red_labels = kwargs["red_labels"]
            self.green_labels = kwargs["green_labels"]
            
        if self.noise_variation == False:
            self.noise_pc = [noise_pc] * self.num_clients

        self.train_x, self.train_y, self.test_x, self.test_y = None, None, None, None
        self.num_labels = None
        
        self.client_data = {}

        # Load the data
        self.load_data()

        # Set up the server data
        self.create_server_data()
        
        # Distribute data
        self.distribute_clients_data(logger)
        
    def load_data(self):
        """Loads the raw pickle data.
        """

        if self.dataset_name == constants.FLOWERS:
            self.file_path = f"Federated/Data/{constants.FLOWERS}/"
        elif self.dataset_name == constants.FEMNIST:
            self.file_path = "Federated/Data/femnist/"
        elif self.dataset_name == constants.CIFAR_10:
            self.file_path = f"Federated/Data/{constants.CIFAR_10}/"
        elif self.dataset_name == constants.CIFAR_100:
            self.file_path = f"Federated/Data/{constants.CIFAR_100}/"
        else:
            assert False, "Dataset not supported in this code!"
            
        with open(self.file_path + "{}_train.pkl".format(self.dataset_name), "rb") as file:
            train = pkl.load(file)
        with open(self.file_path + "{}_test.pkl".format(self.dataset_name), "rb") as file:
            test = pkl.load(file)
        
        self.train_x, self.train_y = torch.Tensor(train[0]), torch.LongTensor(train[1]).view(-1)
        self.test_x, self.test_y = torch.Tensor(test[0]), torch.LongTensor(test[1]).view(-1)
        
        self.labels = torch.unique(self.test_y)
        if self.noise_type == constants.OPENNOISE:
            self.labels = self.green_labels
        self.num_labels = len(self.labels)
       
        
    def create_server_data(self):
        """
        Filter away red data from the test set.
        Server only possesses no-noise version of the test set
        """
        if self.noise_type == constants.OPENNOISE:
            green_idxs = torch.LongTensor([entry for entry, lbl in enumerate(self.test_y) if lbl in self.green_labels])
        else:
            green_idxs = torch.arange(len(self.test_y))
        self.server_data = {"x" : self.test_x[green_idxs],
                            "y" : self.test_y[green_idxs],
                            "green_idxs" : torch.arange(len(green_idxs)),
                            "red_idxs" : []}
        
        
    def distribute_clients_data(self, logger=None):
        """Distributes the dataset across the clients
        1. Distribute according to the dist_strategy -- iid, non-iid, path-non iid. This split happens based on labels
        2. incorporate the noise_type
        3. Assign noise based on the noise_pc
        """
        
        client_data_idxs = None
        
        data_ids = torch.arange(len(self.train_y))
        
        # Distributionn strategy
        if self.dist_strategy == constants.IID:
            client_data_idxs = du.iid_divide(data_ids, self.num_clients)
        elif self.dist_strategy == constants.PATHNONIID:
            client_data_idxs = du.pathological_split(self.train_y, self.num_clients)
        elif self.dist_strategy == constants.NONIID:
            client_data_idxs = du.dirichlet_split(self.train_y, self.num_clients)
        
        clients_x = [ self.train_x[entry] for entry in client_data_idxs]

        # We have to inject noise here
        if self.noise_type == constants.CLOSEDNOISE:
            # This is a list of tuple of client_y and client local red indices
            clients_y_red = [ du.flip_labels(self.train_y[entry], noise_rate=self.noise_pc[idx], y_set=self.labels.tolist()) for idx, entry in enumerate(client_data_idxs) ]
        elif self.noise_type == constants.OPENNOISE:
            clients_y_red = [ du.flip_openlabels(self.train_y[entry], self.green_labels, self.red_labels) for entry in client_data_idxs ]
        elif self.noise_type == constants.ATTRNOISE:
            print("Adding attribute noise to the clients")
            clients_x_red = [ du.add_atttr_noise(clients_x[idx], self.noise_pc[idx]) for idx in range(self.num_clients)]
            clients_x = [ entry[0] for entry in clients_x_red ]
            print("Done adding attribute noise")
            # Add no noise to the labels
            clients_y_red = [ (self.train_y[yidx], torch.LongTensor(nidx[1])) for yidx, nidx in zip(client_data_idxs, clients_x_red) ]
        elif self.noise_type == constants.NONOISE:
            clients_y_red = [ (self.train_y[entry], torch.LongTensor([])) for entry in client_data_idxs ]
        
        for client_id in range(self.num_clients):
            self.client_data[client_id] = {
                "x": clients_x[client_id],
                "y": clients_y_red[client_id][0],
                "green_idxs": du.setdiff1d(torch.arange(len(clients_x[client_id])), clients_y_red[client_id][1]),
                "red_idxs": clients_y_red[client_id][1]
            }
            logger.info(f"Cliet {client_id} has {len(clients_x[client_id])} samples")
            print(f"Cliet {client_id} has {len(clients_x[client_id])} samples")

    def get_server_data(self):
        return self.server_data
    
    def get_client_data(self):
        return self.client_data

        