import numpy as np
np.seterr(all='raise')
import torch

def OrthogonalMP_REG_Parallel_V1(A, b, tol=1E-4, budget=None, positive=True, lam=1, device="cpu"):
    '''approximately solves min_x |x|_0 s.t. Ax=b using Orthogonal Matching Pursuit
    Args:
      A: design matrix of size (d, n)
      b: measurement vector of length d
      tol: solver tolerance
      budget = maximum number of nonzero coefficients (if None set to n)
      positive: only allow positive nonzero coefficients
    Returns:
       vector of length n
    '''
    AT = torch.transpose(A, 0, 1)
    d, n = A.shape
    if budget is None:
        budget = n
    
    # Weights of each data point
    weights = torch.zeros(n, device=device) 
    residue = b.detach().clone()
    orig_residue_norm = residue.norm().item()
    
    global_idxs = torch.arange(n)
    selected_idxs = []
    
    for i in range(budget):
        if residue.norm().item() / orig_residue_norm < tol:
            break
        projections = torch.matmul(AT, residue)  
        
        if positive:
            index = torch.argmax(projections)
        else:
            index = torch.argmax(torch.abs(projections))

        if global_idxs[index] not in selected_idxs:
            selected_idxs.append(global_idxs[index])
            global_idxs = torch.cat((global_idxs[:index], global_idxs[index+1:]))
            AT = torch.cat((AT[:index], AT[index+1:]), dim=0) 
        else:
            assert False, "Why are u selecting an indesx already selected"

        xstar, _ = torch.lstsq(b.view(-1, 1), A[:,selected_idxs])
        xstar = xstar[:len(selected_idxs)]
        residue = b - torch.matmul(A[:,selected_idxs], xstar).T
        residue = residue.squeeze()

    for i, sel_idx in enumerate(selected_idxs):
        weights[sel_idx] = xstar[i]
    
    return selected_idxs, weights



def OrthogonalMP_REG_Parallel_V2(A, b, tol=1E-4, budget=None, positive=True, lam=1, device="cpu"):
    '''approximately solves min_x |x|_0 s.t. Ax=b using Orthogonal Matching Pursuit
    Args:
      A: design matrix of size (d, n)
      b: measurement vector of length d
      tol: solver tolerance
      budget = maximum number of nonzero coefficients (if None set to n)
      positive: only allow positive nonzero coefficients
    Returns:
       vector of length n
    '''
    AT = torch.transpose(A, 0, 1)
    d, n = A.shape
    if budget is None:
        budget = n
    
    # Weights of each data point
    weights = torch.zeros(n, device=device) 
    residue = b.detach().clone()
    orig_residue_norm = residue.norm().item()
    
    chosen  = torch.zeros(AT.shape[0]).bool()
    selected_idxs = []
    global_idxs = torch.arange(n)
    
    for i in range(budget):
        if residue.norm().item() / orig_residue_norm < tol:
            break
        projections = torch.matmul(AT, residue)  
        
        if positive:
            tm_index = torch.argmax(projections)
        else:
            tm_index = torch.argmax(torch.abs(projections))
            
        index = global_idxs[~chosen][tm_index]

        if index not in selected_idxs:
            chosen[index] =  True
            selected_idxs.append(index.item())
        else:
            assert False, "Why are u selecting an indesx already selected"

        xstar, _ = torch.lstsq(b.view(-1, 1), A[:,selected_idxs])
        xstar = xstar[:len(selected_idxs)]
        
        if positive:
            while min(xstar) < 0.0:

                # Added by Lokesh
                if len(xstar) == 1:
                    selected_idxs = []
                    chosen  = torch.zeros(AT.shape[0]).bool()
                    break

                # print("Negative",b.shape,torch.transpose(A_i, 0, 1).shape,x_i.shape)
                argmin = torch.argmin(xstar)
                chosen[selected_idxs[argmin]] =  False
                selected_idxs = selected_idxs[:argmin] + selected_idxs[argmin + 1:]
                
                xstar, _ = torch.lstsq(b.view(-1, 1), A[:,selected_idxs])
                xstar = xstar[:len(selected_idxs)]
                
                '''A_i = torch.cat((A_i[:argmin], A_i[argmin + 1:]),dim=0)  # np.vstack([A_i[:argmin], A_i[argmin+1:]])
                temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + lam * torch.eye(A_i.shape[0], device=device)
                x_i, _ = torch.lstsq(torch.matmul(A_i, b).view(-1, 1), temp)
                x_i = x_i.view(-1)/(A_i.norm(dim=1))'''
        
        if len(selected_idxs) > 0:
            residue = b - torch.matmul(A[:,selected_idxs], xstar).T
            residue = residue.squeeze()

    for i, sel_idx in enumerate(selected_idxs):
        weights[sel_idx] = xstar[i]
    
    return selected_idxs, weights

def OrthogonalMP_REG_Parallel_V1_ED(A, b, tol=1E-4, budget=None, positive=False, lam=1, device="cpu"):
    '''approximately solves min_x |x|_0 s.t. Ax=b using Orthogonal Matching Pursuit
    Args:
      A: design matrix of size (d, n)
      b: measurement vector of length d
      tol: solver tolerance
      budget = maximum number of nonzero coefficients (if None set to n)
      positive: only allow positive nonzero coefficients
    Returns:
       vector of length n
    '''
    AT = torch.transpose(A, 0, 1)
    d, n = A.shape
    if budget is None:
        budget = n
    x = torch.zeros(n, device=device)  # ,dtype=torch.float64)
    resid = b.detach().clone()
    normb = b.norm().item()
    indices = []
    
    chosen  = torch.zeros(AT.shape[0], device=device).bool()
    combine = 5
    if budget < 15:
        combine =1

    for i in range(0,budget,combine):
        if resid.norm().item() / normb < tol:
            break
        
        #print(i,torch.isnan(AT).sum(),torch.isnan(resid).sum())
        projections = (torch.pow(AT - resid, 2).sum(1))
        
        if (~chosen.cpu()).sum() > combine :
            if np.random.rand() < 0.5:#*n/budget :
                p = (projections[~chosen].max() -   projections[~chosen] + 1e-14)/((projections[~chosen].max()+1e-14 - projections[~chosen] ).sum())
                tm_index = np.random.choice((~chosen.cpu()).sum(),combine,replace=False,p=p.detach().cpu().numpy())
            else:
                tm_index = torch.topk(projections[~chosen], combine, largest=False)[1]
        else:
            tm_index = torch.arange((~chosen.cpu()).sum())

        index = torch.arange(AT.shape[0],device=device)[~chosen][tm_index]
        chosen[index] =  True
        indices.extend(index)

        if len(indices) == combine:
            A_i = A[:, index]
            A_i = A[:, index].view(combine, -1)
            temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + lam * torch.eye(A_i.shape[0], device=device)
            x_i = torch.linalg.lstsq(torch.matmul(A_i, b).view(-1, 1), temp).solution.view(-1)
        else:
            A_i = torch.cat((A_i, A[:, index].view(combine, -1)), dim=0)  # np.vstack([A_i, A[:,index]])
            temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + lam * torch.eye(A_i.shape[0], device=device)
            x_i = torch.linalg.lstsq(torch.matmul(A_i, b).view(-1, 1), temp).solution.view(-1)
            if positive:
                while min(x_i) < 0.0:

                    if len(x_i) == 1:
                        indices = []
                        break

                    argmin = torch.argmin(x_i)
                    chosen[indices[argmin]] =  False
                    indices = indices[:argmin] + indices[argmin + 1:]
                    A_i = torch.cat((A_i[:argmin], A_i[argmin + 1:]),
                                    dim=0)  # np.vstack([A_i[:argmin], A_i[argmin+1:]])
                    temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + lam * torch.eye(A_i.shape[0], device=device)
                    x_i, _ = torch.lstsq(torch.matmul(A_i, b).view(-1, 1), temp)
                    x_i = x_i.view(-1)/(A_i.norm(dim=1))
        resid = b - torch.matmul(torch.transpose(A_i, 0, 1), x_i.view(-1,1)).view(-1)
    
    diff = budget - len(indices)

    #print("Points selected", len(indices),len(projections[chosen]))
    if diff > 0:
        #sorted_proj = (torch.abs(projections[~chosen])).sort(descending=True)
        sorted_proj = (projections[~chosen]).sort(descending=False)#True)
        ind = (sorted_proj.indices[:diff]).tolist()
        indices.extend((torch.arange(AT.shape[0])[~chosen][ind]).tolist())
        #x_i = torch.cat((x_i, torch.mean(torch.abs(x_i)) * torch.ones(diff, device=device)))
        x_i = torch.cat((x_i.view(-1), torch.min(x_i) * torch.ones(diff, device=device)))
    

    for i, sel_idx in enumerate(indices):
        x[sel_idx] = x_i[i]
    
    return indices, x
    
