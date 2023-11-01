import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy
from sklearn import metrics

import constants

class Fed_avg(nn.Module):
    """This is the prediction model that needs to be built at the server
    """
    
    def __init__(self, model, optimiser=None, scheduler=None, global_lr=1e-2):
        super(Fed_avg, self).__init__()

        self.model = model.to(constants.CUDA)

        self.global_lr = global_lr

        if optimiser is not None:
            self.optim = optimiser
        else:
            self.optim = optim.SGD(
            [param for param in self.model.parameters() if param.requires_grad],
            lr=self.global_lr, momentum=0.9, weight_decay=5e-4)

        if scheduler is not None:
            self.scheduler = scheduler
        else:
             self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=250)

        # Temp
        # self.scheduler = None
            
        self.criterion_nored = nn.CrossEntropyLoss(reduction='none')
        
        self.__invalidate()
        
        self.g = torch.Generator()
        self.g.manual_seed(0)
    
    def __invalidate(self):
        """Invalidates all the state variables
        """
        self.updates = []
        self.model_state = None
    

    def fit_epochs(self, client_loader, epochs, weights=None):
        """Fits the passed loader on the model for epochs which defaults to 1

        Args:
            client_loader Data loader of the client
            epochs (_type_): 
            weights (_type_, optional): Weights to be used for the cross entropy loss. This is of use only for gradmatch
        Returns:
            Average Loss across all the SGD steps
        """
        self.model.train()
        n_samples = 0
        for client_epoch in range(epochs):
            av_loss = []
            for x, y, batch_idxs in client_loader:
                x = x.to(constants.CUDA).type(torch.float32)
                y = y.to(constants.CUDA)
                n_samples += y.size(0)
                
                self.optim.zero_grad()
                y_pred = self.forward(x)
                loss_vec = self.criterion_nored(y_pred, y)
                
                if weights is not None:
                    weights = weights.to(constants.CUDA, dtype=torch.float32)
                    loss = (loss_vec.T @ weights[batch_idxs])/torch.sum(weights[batch_idxs])
                else:
                    loss = loss_vec.mean()    
                
                av_loss.append(loss.cpu().item())
                loss.backward()
                self.optim.step()
        
        return torch.mean(torch.Tensor(av_loss))

    def save_model_state(self):
        """
        saves the model state in a class variable
        It also saves the model parameters to compute the difference later
        """
        assert self.model_state == None, "When u try save model state, better have it cleared"
        self.model_state = deepcopy(self.state_dict())

    def load_model_state(self):
        """
        This will initialize the model with self.model_state variable
        """
        assert self.model_state != None, "Why is the model state not clear while saving the pred model"
        self.load_state_dict(self.model_state)


    def federated_avg(self, clients_ds_dict, clients_weights=None, epoch=1, batch=32, tst_datset=None, verbose=False):
        """Runs one communication round of federated averaging

        Args:
            clients_ds_dict (_type_): dictionary of clients datasets
            epoch (_type_): Number of local epochs to be run on each clients dataset
            batch (_type_): batch size to be used in local epoch
            clients_weights (_type_, optional): weights of data points to bve used while computing the loss
            tst_datset (_type_, optional): Test dataset to be used to record the test performance
        """
        assert len(self.updates) == 0, "Updates should be flushed before fed_avg"
        
        self.model.train()
            
        num_clients_shared = len(clients_ds_dict.keys())

        self.save_model_state()
        
        tst_loader = DataLoader(tst_datset, batch_size=batch, shuffle=False)
        
        for client_id, client_ds in clients_ds_dict.items():
            
            if verbose:
                assert tst_loader is not None, "If u want verbose, admit the pain of passing a testloader"
                client_perf_bef = self.evaluate_model(tst_loader)
            
            client_loader = DataLoader(client_ds, batch_size=batch, shuffle=True,generator=self.g)
            cweights = None
            if clients_weights is not None:
                cweights = clients_weights[client_id]

            self.fit_epochs(client_loader, epoch, cweights)

            if verbose:
                client_perf_aft = self.evaluate_model(tst_loader)
            
            self.updates.append(deepcopy(self.state_dict()))

            self.load_model_state()
            
            if verbose:
                print(f"Client: {client_id}; before={client_perf_bef}; client_perf_aft={client_perf_aft}")
    
        self.model_state = None

        if self.scheduler is not None:
            self.scheduler.step()

        assert len(self.updates) == num_clients_shared, "There is something phishy with fed_avg"
        
        weights = (1 / num_clients_shared) * torch.ones(num_clients_shared, device=constants.CUDA)

        target_state_dict = self.state_dict()
        
        for key in target_state_dict:
            if target_state_dict[key].data.dtype == torch.float32:
                target_state_dict[key].data.fill_(0.)
                for idx, client_updates in enumerate(self.updates):
                    target_state_dict[key].data += weights[idx] * client_updates[key].data.clone()
        self.updates = []

        if verbose:
            print(f"Performance after federated averaging: {self.evaluate_model(tst_loader)}")
        
    def forward(self, input_x, last=False, freeze=False):
        """Runs forward on the federated global model

        Args:
            input_x (_type_): _description_
            last (bool, optional): _description_. Defaults to False.
            freeze (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        out, feat = self.model(input_x)
        if last:
            return out, feat
        else:
            return out

        
    def evaluate_model(self, loader:DataLoader):
        """Evaluates the model on the passed loader

        Args:
            loader (DataLoader): _description_

        Returns:
            _type_: _description_
        """
        tst_correct =0
        t_size = 0
        self.model.eval()
        with torch.no_grad():
            for x, y, _ in loader:
                x = x.to(constants.CUDA).type(torch.float32)
                y = y.to(constants.CUDA)
                preds = torch.softmax(self.forward(x), dim=1)
                _, predicted = preds.max(1)
                tst_correct += predicted.eq(y).sum().item()
                t_size += y.size(0)
        return tst_correct*100/t_size
 