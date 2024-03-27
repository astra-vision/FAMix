import torch

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.reshape(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.reshape(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def divide_in_patches(feat,n):
    '''
    n is the number of patches in one dimension (H or W)
    '''
    size = feat.size()
    assert (len(size) == 4)
    N,C,H,W = size
    
    return torch.stack([feat[:,:,i*(H//n):(i+1)*(H//n),j*(W//n):(j+1)*(W//n)] for i in range(n) for j in range(n)])

