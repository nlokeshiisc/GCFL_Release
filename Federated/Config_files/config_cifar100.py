import constants

config = {
    "server_specs" : {
        "lr" : 1e-2,
        "client_select_strategy" : "all", #"random", 
        "pred_model_type" : constants.CNN, #constants.NN, # "CNN", "mobilenet"
        constants.HID_DIM : [100,100], # pred_model specs [100,100]
        "fed_batch_size": 32, # batch size to be used while running federated learning on the sampled data from clients
        constants.SERVER_EPOCHS: 1,
        constants.SERVER_VALSIZE: 0.2
        },

    "gradmatch_specs" : {
        "sampling_budget": 0.1, # in terms of percentages now.
        "gm_valgrad_reg": True,
        "warm_start" : 2,
    },

    "client_specs": {
        "subset_sel_strategy": "gm", # gm, random, full, skyline
        "subset_sel_rnd": 10, # 5/10
        "misc": {
            # Put everything else here. For instance controlled_random may need us to specify the green/red ratio in the client samples
        }
    },
    
    "dataset_specs": {
        "dataset_name" : constants.CIFAR_100, # mnist
        "dist_strategy": constants.NONIID, # non-iid, iid
        "noise_type" : constants.CLOSEDNOISE, # noise, labels
        "noise_pc" : 0.4, # This is a scalar only when the noise_variation is False
        "noise_variation" : False, 
        "green_labels" : [0,1,2,3,4], # These are the target labels
        "red_labels" : [5,6,7,8,9], # These are the irrelevant labels
    },
    
    "federated_specs" : {
        "num_clients" : 10,
        "round_clients" : 10,
        "num_rounds" : 251,
    },
    
    constants.LOG_FILE: "try.log",
    constants.GPU_ID: 2,
    constants.SKYLINE:False
}

# Code taken from https://github.com/omarfoq/FedEM