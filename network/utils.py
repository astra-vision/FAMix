import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.stats import *
import random
from torch.nn.functional import unfold



class _Segmentation(nn.Module):
    def __init__(self, backbone,classifier):
        super(_Segmentation, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
      
    def forward(self, x, transfer=False,mix=False,most_list=None,saved_params=None,activation=None,s=0):
        input_shape = x.shape[-2:]
        features = {}
        features['low_level'] = self.backbone(x,trunc1=False,trunc2=False,
           trunc3=False,trunc4=False,get1=True,get2=False,get3=False,get4=False)
        d2 ,d3 = features['low_level'].shape[2], features['low_level'].shape[3]

        if transfer:

            mean, std = calc_mean_std(features['low_level'])
            self.size = features['low_level'].size()

            mu_t_f1 = torch.zeros([8,256,d2,d3])
            std_t_f1 = torch.zeros([8,256,d2,d3])
            h=0
            w=0

            self.patches = unfold(features['low_level'], kernel_size=64, stride=64).permute(-1,0,1)
            self.patches = self.patches.reshape(self.patches.shape[0],self.patches.shape[1],256,d2//3,d3//3)

            means_orig = torch.zeros([8,256,d2,d3])
            stds_orig = torch.zeros([8,256,d2,d3])

            for i in range(3*3):
                mean , std = calc_mean_std(self.patches[i])
            
                means_orig[:,:,h:h+d2//3,w:w+d3//3] = mean.expand((8,256,d2//3,d3//3))
                stds_orig[:,:,h:h+d2//3,w:w+d3//3] =std.expand((8,256,d2//3,d3//3))
                w+=d2//3
                if (i+1)%3 == 0 :
                    w=0
                    h+=d3//3

                self.patches[i] = (self.patches[i] - mean.expand(self.patches[i].size()) ) / std.expand(self.patches[i].size())
            features_low_norm = torch.cat([torch.cat([self.patches[3*i+j] for i in range(3)],dim=2) for j in range(3)],dim=3)
            
            h=0
            w=0
            
            for j,most in enumerate(most_list):  #len(most_list)=div*div   
                for k,el in enumerate(most):  #len(most)=B

                    if not saved_params[str(el)+'_mu']:
                        idx = random.choice([idxx for idxx in range (len(saved_params['255_mu']))])
                        mu_t = saved_params['255_mu'][idx]
                        std_t = saved_params['255_std'][idx]
                    else:
                        #orig
                        idx = random.choice([idxx for idxx in range (len(saved_params[str(el)+'_mu']))])
                        mu_t = saved_params[str(el)+'_mu'][idx]
                        std_t = saved_params[str(el)+'_std'][idx]

                    mu_t_f1[k,:,h:h+features['low_level'].shape[2]//3,w:w+features['low_level'].shape[3]//3]  = mu_t.expand((256,d2//3,d3//3))
                    std_t_f1[k,:,h:h+d2//3,w:w+d3//3] = std_t.expand((256,d2//3,d3//3))
                w+=d2//3
                if (j+1)%3==0:
                    w=0
                    h+=d3//3
            if not mix:
                features['low_level'] = (std_t_f1.to('cuda') * features_low_norm + mu_t_f1.to('cuda'))
            else:

                mu_mix = s * means_orig.to('cuda') + (1-s) *  mu_t_f1.to('cuda')
                std_mix = s * stds_orig.to('cuda') + (1-s) *  std_t_f1.to('cuda')
                features['low_level'] = (std_mix.expand(self.size) * features_low_norm + mu_mix.expand(self.size))
            features['low_level'] = activation(features['low_level'])
           
        features['out'] = self.backbone(features['low_level'],trunc1=True,trunc2=False,
            trunc3=False,trunc4=False,get1=False,get2=False,get3=False,get4=True)
    
        x = self.classifier(features)
        output = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return output, features