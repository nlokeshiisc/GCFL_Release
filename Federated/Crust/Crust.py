import numpy as np
from torch.autograd import grad

def estimate_grads(trainval_loader, model, criterion, device):
    # switch to train mode
    model.train()
    all_grads = []
    all_targets = []
    all_preds = []
    for i, (input, target, idxs) in enumerate(trainval_loader):

        input = input.to(device)
        all_targets.append(target)
        target = target.to(device)
        
        output, feat = model(input,last=True)
        loss = criterion(output, target).mean()
        
        est_grad = grad(loss, feat)
        all_grads.append(est_grad[0].detach().cpu().numpy())
        
    all_grads = np.vstack(all_grads)
    all_targets = np.hstack(all_targets)
    
    return all_grads, all_targets