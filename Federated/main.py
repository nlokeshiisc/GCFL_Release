import constants
import os
import argparse
import importlib.util
import warnings
warnings.filterwarnings("ignore")
from coordinator import Coordinator
import time
from utils import set_seed, set_cuda_device, dict_print, plot_fed_pred_accs
import logging
from pathlib import Path
import torch

# Initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Use the correct argument', 
                    default="Federated/Config_files/config.py")
# Read arguments from the command line
args, unknown_args = parser.parse_known_args()
#File name without extension
file_name = os.path.splitext(args.config)[0]
file_name = file_name.split("/")[0]
spec = importlib.util.spec_from_file_location(file_name, args.config)
# pass above spec object to module_from_spec function to get custom python module from above module spec.
config_module = importlib.util.module_from_spec(spec)
# load the module with the spec.
spec.loader.exec_module(config_module)
config = config_module.config
# print(config)

#overriding config. This can be used to run jobs from a script file. Pass the intended modifications as command line arguments
for override_config in unknown_args:
    parts = override_config.split(":")
    key = parts[0]
    value = parts[1]
    if "." in key:
        key_parts = key.split(".")
        primary_key = key_parts[0]
        secondary_key = key_parts[1]
        try:
            config[primary_key][secondary_key] = eval(value)
        except:
           config[primary_key][secondary_key] = value
    else:
        config[key] = value
print(config, flush=True)
Path("Federated/logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=f"Federated/logs/{str(config[constants.LOG_FILE])}",
                        format='%(asctime)s :: %(filename)s:%(funcName)s :: %(message)s',
                        filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.info(dict_print(config))

# Run 
set_seed()
coordinator = Coordinator(config, logger)

num_comm_rounds = config["federated_specs"]["num_rounds"]
accs = []

total_memory = 0

for comm_round in range(num_comm_rounds):
    start_time = time.time()
    max_gpu_usage = torch.cuda.memory_allocated(constants.CUDA)
    fed_acc = coordinator.perform_round(comm_round, logger)
    accs.append(fed_acc)
    total_memory = max(torch.cuda.memory_allocated(constants.CUDA) - max_gpu_usage,total_memory)
    print("Round {} Time {}".format(comm_round, time.time() - start_time), flush=True)
    print("Round {} GPU Memory {}".format(comm_round, total_memory), flush=True)
    logger.info("*****"*10 + f"Round {comm_round} Time {time.time() - start_time}" + "*****"*10)

# plot_fed_pred_accs(fed=accs, save_name=str(config[constants.LOG_FILE])[:-4])