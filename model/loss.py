# coding=utf-8
import torch
from torch.nn import functional as F

#from model.utils import euclidean_dist


def contrastive_loss(encoded, encoded_aug, temp=0.07):
    assert(encoded.size()==encoded_aug.size())

    cl_total = 0
    encoded_norm = F.normalize(encoded, p=2, dim = 1)
    encoded_aug_norm = F.normalize(encoded_aug, p=2, dim=1)

    aug_batch = torch.cat([encoded_norm, encoded_aug_norm], dim=0)
    n = aug_batch.size()[0]

    for i in range(n):
        # Get the index of the pair
        j = (i+(n//2)) % n 

        cl_i_num = torch.exp(torch.dot(aug_batch[i,:],aug_batch[j,:])/temp)
        cl_i_den = 0

        for k in range(n):
            if(i!=k):
                cl_i_den += torch.exp(torch.dot(aug_batch[i,:],aug_batch[k,:])/temp)
        
        cl_total += -torch.log(cl_i_num/cl_i_den)
        print(cl_i_num/cl_i_den)
        print(cl_total)
    return cl_total

batch = torch.ones([4,10])
batch_aug = torch.ones([4,10])*2

print(contrastive_loss(batch,batch_aug))


def loss_task(encoded, prototypes, target, criterion='dist', encoded_augment=None):
    """Calculate loss.
    criterion: NNLoss - assign to closest prototype and calculate NNLoss
         dist - loss is distance to prototype that example needs to be assigned to
                and -distance to prototypes from other class
    """
    
    uniq = torch.unique(target, sorted=True)
    class_idxs = list(map(lambda c: target.eq(c).nonzero(), uniq))
    
    # prepare targets so they start from 0,1
    for idx,v in enumerate(uniq):
        target[target==v]=idx
    
    dists = euclidean_dist(encoded, prototypes)
    
    if criterion=='NNLoss':
       
        loss = torch.nn.NLLLoss()
        log_p_y = F.log_softmax(-dists, dim=1)
        
        loss_val = loss(log_p_y, target)
        _, y_hat = log_p_y.max(1)
        
    
    elif criterion=='dist':
        
        loss_val = torch.stack([dists[idx_example, idx_proto].mean(0) for idx_proto,idx_example in enumerate(class_idxs)]).mean()
        #loss_val1 = loss_val1/len(embeddings) 
        y_hat = torch.max(-dists,1)[1]
        
    acc_val = y_hat.eq(target.squeeze()).float().mean()    
    
    if(encoded_augment is not None):
        loss_val += contrastive_loss(encoded, encoded_augment)

    return loss_val, acc_val

def loss_test_nn(encoded, prototypes):
    dists = euclidean_dist(encoded, prototypes)
    min_dist = torch.min(dists, 1)
    
    y_hat = min_dist[1]
    args_uniq = torch.unique(y_hat, sorted=True)
    args_count = torch.stack([(y_hat==x_u).sum() for x_u in args_uniq])
    print(args_count)
    
    loss = torch.nn.NLLLoss()
    log_p_y = F.log_softmax(-dists, dim=1)
    print(log_p_y.shape)
        
    loss_val = loss(log_p_y, y_hat)
    _, y_hat = log_p_y.max(1)
    
    return loss_val, args_count


def loss_test_basic(encoded, prototypes):
    dists = euclidean_dist(encoded, prototypes)
    min_dist = torch.min(dists, 1)
    
    y_hat = min_dist[1]
    args_uniq = torch.unique(y_hat, sorted=True)
    args_count = torch.stack([(y_hat==x_u).sum() for x_u in args_uniq])
    #print(args_count)
    
    min_dist = min_dist[0] # get_distances
    
    #thr = torch.stack([torch.sort(min_dist[y_hat==idx_class])[0][int(len(min_dist[y_hat==idx_class])*0.9)] for idx_class in args_uniq])
    #loss_val = torch.stack([min_dist[y_hat==idx_class][min_dist[y_hat==idx_class]>=thr[idx_class]].mean(0) for idx_class in args_uniq]).mean()
    
    loss_val = torch.stack([min_dist[y_hat==idx_class].mean(0) for idx_class in args_uniq]).mean()
    
    #loss_val,_ = loss_task(encoded, prototypes, y_hat, criterion='dist') # same
    
    return loss_val, args_count

def loss_test(encoded, prototypes, tau, encoded_augment):
    #prototypes = torch.stack(prototypes).squeeze() 
    loss_val_test, args_count = loss_test_basic(encoded, prototypes)
    
    if tau>0:
        dists = euclidean_dist(prototypes, prototypes)
        nproto = prototypes.shape[0]
        loss_val2 = - torch.sum(dists)/(nproto*nproto-nproto)
        
        loss_val_test += tau*loss_val2
        
    if(encoded_augment):
        loss_val_test += contrastive_loss(encoded, encoded_augment)
    return loss_val_test, args_count

def reconstruction_loss(decoded, x):
    loss_func = torch.nn.MSELoss()
    loss_rcn = loss_func(decoded, x)
    #print('Reconstruction {}'.format(loss_rcn))
    
    return loss_rcn
    
    
