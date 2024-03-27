import torch
import torch.nn as nn
from torch.nn.functional import unfold
from utils.stats import calc_mean_std

class PPIN(nn.Module):
    def __init__(self,content_feat,div=3,ind=[]):
        super(PPIN,self).__init__()
        self.ind = ind
        self.div = div
        self.content_feat = content_feat.clone().detach() #(B,C,H,W)
        self.B = self.content_feat.shape[0]
        self.C = self.content_feat.shape[1]

        self.patches = unfold(self.content_feat, kernel_size=64, stride=64).permute(-1,0,1).reshape(-1,256,64,64)
        self.patches = self.patches[ind] #(len(ind),C,H/div,W/div)
        
        self.style_mean = torch.zeros(len(ind),self.C,1,1) #(len(ind),C,1,1)
        self.style_std =  torch.zeros(len(ind),self.C,1,1) #(len(ind),C,1,1)
      
        for i in range(len(self.ind)):

            mean , std = calc_mean_std(self.patches[i].unsqueeze(0))
            self.patches[i] = (self.patches[i] - mean.expand(self.patches[i].unsqueeze(0).size()) ) / std.expand(self.patches[i].unsqueeze(0).size())
            self.style_mean[i] = mean
            self.style_std[i] = std

        self.size = self.patches[0].size()   #(C,H/div,W/div)

        self.style_mean = nn.Parameter(self.style_mean, requires_grad = True) # (len(ind),C,1,1)
        self.style_std = nn.Parameter(self.style_std, requires_grad = True) #(len(ind),C,1,1)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self):
        
        patches_prime = torch.zeros_like(self.patches.clone().detach()) # (len(ind),C,H/div,W/div)
        
        for i in range(len(self.ind)):
            patches_prime[i] = self.patches[i] * self.style_std[i].expand(self.size)  + self.style_mean[i].expand(self.size)
        
        patches_prime = self.relu(patches_prime)

        return patches_prime