
import time 
from global_data import Global_data
from client import Client
from server import Server
import constants
import torch
import logging

class Coordinator:
    
    def __init__(self, config, logger):
        self.config = config
# %% Process the Dataset
        ds_kwargs = {}
        self.dataset_name = self.config["dataset_specs"]["dataset_name"]
        self.dist_strategy = self.config["dataset_specs"]["dist_strategy"]
        self.noise_type = self.config["dataset_specs"]["noise_type"]
        self.noise_pc = self.config["dataset_specs"]["noise_pc"]
        self.noise_variation = self.config["dataset_specs"]["noise_variation"]
        
        self.green_labels, self.red_labels = None, None
        if self.noise_type == constants.VARNOISE:
            assert type(self.noise_pc) == list and len(self.noise_pc) > 1, "We need a list of noise at each client for variationj noise"
            
        if self.noise_type == constants.OPENNOISE:
            self.green_labels = self.config["dataset_specs"]["green_labels"]
            self.red_labels = self.config["dataset_specs"]["red_labels"]
            ds_kwargs["green_labels"] = self.green_labels
            ds_kwargs["red_labels"] = self.red_labels
            
        self.num_clients = self.config["federated_specs"]["num_clients"]
        self.num_rounds = self.config["federated_specs"]["num_rounds"]
        self.round_clients = self.config["federated_specs"]["round_clients"]

        self.global_info = Global_data(dataset_name=self.dataset_name, 
                                       dist_strategy = self.dist_strategy,
                                       num_clients=self.num_clients,
                                       noise_type=self.noise_type,
                                       noise_pc=self.noise_pc,
                                       noise_variation=self.noise_variation,
                                       logger=logger,
                                       **ds_kwargs)
        
        # %% 
        # Set up the server
        self.client_ids = torch.arange(self.num_clients)
        self.client_sel_strategy = self.config["server_specs"]["client_select_strategy"]
        self.pred_model_arch =  self.config["server_specs"][constants.HID_DIM]
        self.server_pred_model_type = self.config["server_specs"]["pred_model_type"]
        self.server_lr = self.config["server_specs"]["lr"]
        self.fed_batch_size = self.config["server_specs"]["fed_batch_size"]
        self.fed_epochs = self.config["server_specs"][constants.SERVER_EPOCHS]
        self.server_valsize = self.config["server_specs"][constants.SERVER_VALSIZE]
        self.setup_server(logger)

        # %% 
        # Set up the clients

        self.subset_strategy = self.config["client_specs"]["subset_sel_strategy"]
        self.subset_sel_rnd = self.config["client_specs"]["subset_sel_rnd"]

        self.sampling_budget = self.config["gradmatch_specs"]["sampling_budget"]
        self.gm_valgrad = self.config["gradmatch_specs"]["gm_valgrad_reg"]
        self.warm_start = self.config["gradmatch_specs"]["warm_start"]
        self.skyline = self.config[constants.SKYLINE]

        # For full, samplking budget is anyways redundant. So let us make it correct
        if self.client_sel_strategy == "full":
            self.sampling_budget = 1.
        if self.client_sel_strategy == "random-warm":
            assert self.sampling_budget != 1., "for random-warm, we need a sampling budget less than 1. o.w. set use sampling strategy full"
        
        self.setup_clients(logger)
                               
       
    def setup_server(self, logger:logging.Logger):
        server_data = self.global_info.server_data

        self.server = Server(X=server_data["x"], y=server_data["y"],
                            dataset_name=self.dataset_name, 
                            model_type=self.server_pred_model_type,
                            model_arch=self.pred_model_arch,
                            batch_size=self.fed_batch_size,
                            global_lr=self.server_lr,
                            client_ids=self.client_ids,
                            num_rounds=self.num_rounds,
                            fed_epochs=self.fed_epochs,
                            val_size = self.server_valsize,
                            logger=logger)


    def setup_clients(self, logger):
        self.clients = []
        
        for client_id in range(self.num_clients):
            client_data = self.global_info.client_data[client_id]

            client = Client(client_id=client_id,
                            X=client_data["x"], y=client_data["y"],
                            dataset_name=self.dataset_name, 
                            green_idxs=client_data["green_idxs"], red_idxs=client_data["red_idxs"],
                            server_pred_model=self.server.model,
                            sampling_strategy=self.subset_strategy, 
                            server_valid_ds=self.server.valid_ds,
                            sampling_pc=self.sampling_budget,
                            num_labels=self.global_info.num_labels,
                            gm_valgrad_reg=self.gm_valgrad, 
                            subset_sel_rnd=self.subset_sel_rnd,
                            warm_start=self.warm_start,
                            logger=logger)

            self.clients.append(client)
            
        print("Added {} clients in the coordinator".format(self.num_clients))
        
        
    def perform_round(self, round_num, logger):
        # Get the clients for this round
        round_clients = self.server.get_round_clients(self.client_sel_strategy, self.round_clients)

        sel_clients_data = {}
        for sel_client_id in round_clients:
            dataset, wts = self.clients[sel_client_id].get_round_data(round_num, self.server.get_cls_state(), logger) 
            
            if dataset is not None:
                
                sel_clients_data[sel_client_id] = {
                    constants.FED_CLIDATA: dataset,
                    constants.FED_CLIWTS: wts
                }
        
        round_acc = self.server.update_fed_model(sel_clients_data, round_num=round_num, 
                                    fed_epochs=self.fed_epochs, fed_batch=self.fed_batch_size)

        print(f"Round {round_num}; fed_acc: {round_acc}", flush=True)

        if logger is not None:
            logger.info(f"Round {round_num}; fed_acc: {round_acc}")
        return round_acc