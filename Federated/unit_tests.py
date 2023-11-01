from sympy import glsl_code
from global_data import Global_data
import constants as constants
import logging
import models
from fed_avg import Fed_avg
from data import CustomDataset
import torch
import torch.nn as nn
from data_utils import setdiff1d
import numpy as np
from torch.utils.data import DataLoader
from cords.gradmatchstrategy import GradMatchStrategy
import copy 
from utils import set_seed

import pickle as pkl


logging.basicConfig(filename=f"Federated/logs/unit_tests.log",
                        format='%(asctime)s :: %(filename)s:%(funcName)s :: %(message)s',
                        filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

global_info = Global_data(dataset_name=constants.LETTER, 
                            dist_strategy = constants.NONIID,
                            num_clients=10,
                            noise_type=constants.CLOSEDNOISE,
                            noise_pc=0.4,
                            noise_variation=False,
                            logger=logger)

clients_data = global_info.get_client_data() # This contains the list of datasets for 10 clients

print(len(clients_data))

model_args = {
    constants.HID_DIM: [100, 100]
}
model = models.get_pred_model(model_type=constants.NN, dataset_name=constants.LETTER,
                        num_classes=global_info.num_labels, num_features=global_info.train_x.shape[1], 
                        **model_args)

print(model)

##FED_AVGERAGE CODE
client_dataset ={}
for k,v in clients_data.items():
    client_dataset[k] = CustomDataset(v['x'], v['y'],[],[])
    
v = global_info.get_server_data()
test_data = CustomDataset(v['x'], v['y'],[],[])

fa = Fed_avg(model)

# for i in range(500):
#     fa.feverated_avg(client_dataset, 1, 32, tst_datset=test_data, verbose=True)


# ##GRADMATCH CODE
first =True
for k,v in clients_data.items():
    if first:
        x = v['x']
        y = v['y']
        green_idxs = v['green_idxs']
        red_idxs = v['red_idxs']
        first = False
    else:
        green_idxs = torch.cat((green_idxs,y.size(0)+v['green_idxs']),dim=0)
        red_idxs = torch.cat((red_idxs,y.size(0)+v['red_idxs']),dim=0)
        
        x = torch.cat((x,v['x']),dim=0)
        y = torch.cat((y,v['y']),dim=0)

train_data = CustomDataset(x, y,green_idxs,red_idxs)

g = torch.Generator()
g.manual_seed(0)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, generator=g)

v = global_info.get_server_data()
nt = int(len(v['x'])*0.9)
nv = len(v['x']) -nt
X_te, X_va = torch.split(v['x'], [nt,nv])
y_te, y_va = torch.split(v['y'], [nt,nv])

val_loader = DataLoader(CustomDataset(X_va, y_va,[],[]), batch_size=32, shuffle=False, generator=g)
tst_loader = DataLoader(CustomDataset(X_te, y_te,[],[]), batch_size=32, shuffle=False, generator=g)

gm  = GradMatchStrategy(train_loader, val_loader, copy.deepcopy(model.to(constants.CUDA)),  nn.CrossEntropyLoss(reduction='none'),\
    1e-2,constants.CUDA, global_info.num_labels, True,'PerClassPerGradient',valid=True, lam=0.5)

fa = Fed_avg(model)


# %% 
#Gradmatch
set_seed()
for i in range(100):
    if i % 10 == 0:
        if i == 0:
            gm_subset_idxs = torch.LongTensor(np.random.choice(len(y), int(0.1 * len(y)), replace=False))
        else:
            gm_subset_idxs, gm_subset_wts = gm.select(int(y.size(0)*0.1), copy.deepcopy(model.state_dict()))
        sub_loader = DataLoader(CustomDataset(x[gm_subset_idxs], y[gm_subset_idxs],None,None),batch_size=32, shuffle=True)
        logger.info(f"Total green points{str(len(green_idxs))} GM selected {str(len(gm_subset_idxs))} Budget{str(int(y.size(0)*0.1))} Green points selected {str(len(setdiff1d(torch.tensor(gm_subset_idxs), red_idxs)))}")
    client_perf_bef = fa.evaluate_model(tst_loader)
    fa.fit_epochs(sub_loader, 1, weights=None)
    client_perf_aft = fa.evaluate_model(tst_loader)
    logger.info(f"Epoch={i}; accuracy={client_perf_aft}")

#with open('tmp_data.pkl','wb') as file:
#    pkl.dump((x,y,CustomDataset(X_te, y_te,[],[])),file)
    
# with open('tmp_data.pkl','rb') as file:
    
#     x,y, tst_dataset = pkl.load(file)       

#tst_loader = DataLoader(tst_dataset, batch_size=32, shuffle=False, generator=g)    
# %%
# Random performance
# set_seed()
# for i in range(100):
#     #set_seed(i)
#     if i % 10 == 0:
#         #set_seed(i)
#         gm_subset_idxs = torch.LongTensor(np.random.choice(y.size(0), int(0.1 * len(y)), replace=False))
#         sub_loader = DataLoader(CustomDataset(x[gm_subset_idxs], y[gm_subset_idxs],None,None),batch_size=32, shuffle=True, generator=g)
#         logger.info(f"Total green points{str(len(green_idxs))} GM selected {str(len(gm_subset_idxs))} Budget{str(int(y.size(0)*0.1))} Green points selected {str(len(setdiff1d(torch.tensor(gm_subset_idxs), red_idxs)))}")
#     client_perf_bef = fa.evaluate_model(tst_loader)
#     fa.fit_epochs(sub_loader, 1, weights=None)
#     client_perf_aft = fa.evaluate_model(tst_loader)

#     logger.info(f"Epoch={i}; Accuracy={client_perf_aft}")