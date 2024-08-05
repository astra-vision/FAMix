import pickle
import os
import clip
import torch
import network
import torch.nn as nn
from datasets import cityscapes
import numpy as np
import random
import argparse
from utils.get_dataset import get_dataset
from torch.utils import data
from utils.freeze import freeze_all
from torch.nn.functional import unfold
from utils.PPIN import PPIN 

from torch.utils.tensorboard import SummaryWriter


def compose_text_with_templates(text: str, templates) -> list:
    return [template.format(text) for template in templates]

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

random_styles = ['Ethereal Mist style',
'Cyberpunk Cityscape style',
'Rustic Charm style',
'Galactic Fantasy style',
'Pastel Dreams style',
'Dystopian Noir style',
'Whimsical Wonderland style',
'Urban Grit style',
'Enchanted Forest style',
'Retro Futurism style',
'Monochrome Elegance style',
'Vibrant Graffiti style',
'Haunting Shadows style',
'Steampunk Adventures style',
'Watercolor Serenity style',
'Industrial Chic style',
'Cosmic Voyage style',
'Pop Art Popularity style',
'Abstract Symphony style',
'Magical Realism style',
]


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    parser.add_argument("--data_root", type=str, default='/datasets_master/gta5', help="path to dataset")
    parser.add_argument("--save_dir", type=str, help= "path for learnt parameters saving")
    parser.add_argument("--dataset", type=str, default='gta5', choices=['cityscapes','gta5','synthia'], help='Name of dataset')
    parser.add_argument("--crop_size", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=16, help='batch size (default: 16)')
    parser.add_argument("--lr", type=float, default=1, help='optimization step')
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet_clip', choices=available_models, help='model name')
    parser.add_argument("--BB", type=str, default = 'RN50', help= "backbone name" )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='weight decay (default: 1e-4)')
    parser.add_argument("--total_it", type = int, default =100, help= "total number of optimization iterations")
    # learn statistics
    parser.add_argument("--resize_feat",action='store_true',default=False, help="resize the features map to the dimension corresponding to CLIP")
    parser.add_argument("--div", type=int, default=3)
    # random seed
    parser.add_argument("--random_seed", type=int, default=1, help="random seed (default: 1)")
    #data augmentation
    parser.add_argument("--data_aug", action='store_true',default=False)
   
    return parser

def main(random_styles=random_styles):


    opts = get_argparser().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    # INIT
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    #Load data
    train_dst,val_dst = get_dataset(opts.dataset,opts.data_root,opts.crop_size,data_aug=opts.data_aug)
    train_loader = data.DataLoader( train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=4, drop_last=True)  # drop_last=True to ignore single-image batches.
    print("Dataset: %s, Train set: %d, Val set: %d" % (opts.dataset, len(train_dst), len(val_dst)))

    #load CLIP
    model = network.modeling.__dict__['deeplabv3plus_resnet_clip'](num_classes=19,BB= opts.BB,OS=32)
    freeze_all(model)

    clip_model, preprocess = clip.load(opts.BB, device, jit=False)
    torch.autograd.set_detect_anomaly(True)

    cur_itrs = 0

    writer = SummaryWriter()
    
    if not os.path.isdir(opts.save_dir):
        os.mkdir(opts.save_dir)

    if opts.resize_feat:
        t1 = nn.AdaptiveAvgPool2d((56,56))
    else:
        t1 = lambda x:x


    #Define the dictionary of stats to save
    stats = {'road_mu': [],
            'sidewalk_mu':[],
			'building_mu':[],
			'wall_mu':[],
			'fence_mu':[],
			'pole_mu':[],
			'traffic light_mu':[],
			'traffic sign_mu':[],
			'vegetation_mu':[],
			'terrain_mu':[],
			'sky_mu':[],
			'person_mu':[],
			'rider_mu':[],
			'car_mu':[],
			'truck_mu':[],
			'bus_mu':[],
			'train_mu':[],
			'motorcycle_mu':[],
			'bicycle_mu':[],
            '255_mu':[],

            'road_std': [],
            'sidewalk_std':[],
			'building_std':[],
			'wall_std':[],
			'fence_std':[],
			'pole_std':[],
			'traffic light_std':[],
			'traffic sign_std':[],
			'vegetation_std':[],
			'terrain_std':[],
			'sky_std':[],
			'person_std':[],
			'rider_std':[],
			'car_std':[],
			'truck_std':[],
			'bus_std':[],
			'train_std':[],
			'motorcycle_std':[],
			'bicycle_std':[],
            '255_std':[],
    }


    for i,(images, labels) in enumerate(train_loader):
            print("i = ",i)
  
            f1 = model.backbone(images.to(device),trunc1=False,trunc2=False,
            trunc3=False,trunc4=False,get1=True,get2=False,get3=False,get4=False)
              
            #the target text is determined by the most frequent class in the corresponding crop
            labels_ = labels.unsqueeze(1)  # (B,1,768,768)
            if labels_.type() != 'torch.FloatTensor':
                labels_ = labels_.to(torch.float32)

            lbl_patches = unfold(labels_, kernel_size=256, stride=256).permute(-1,0,1).reshape(-1,1,256,256) ## (div*div*B,1,H/div,W/div)

            most_list = []
            
            for j in range(lbl_patches.shape[0]):
                
                most_list.append(cityscapes.Cityscapes.name(int(torch.mode(torch.flatten(lbl_patches[j,:,:,:])).values)) if torch.mode(torch.flatten(lbl_patches[j,:,:,:])).values != 255 else 255)

            unique = list(set(most_list))
            ind = []
            for s in unique:
                ind.append(most_list.index(s)) #list of indices for first occurence of each unique element
                
            # most_list , ind = lbl_patches.flatten(-2,-1).mode(dim=-1)

            text_target = torch.zeros((len(ind),1024)).type(torch.float32).to(device) # (len(ind),1024) 1024 is the dim of CLIP latent space (RN50)


            for j,k in enumerate(unique):
              
                if k==255:
                    target = "photo"
                else:
                    target =random.choice(random_styles) + ' ' + k


                target = compose_text_with_templates(target, imagenet_templates)

                tokens = clip.tokenize(target).to(device)

                text_target[j] = clip_model.encode_text(tokens).mean(axis=0, keepdim=True).detach()
                text_target[j] /= text_target[j].norm(dim=-1, keepdim=True)

            #optimize mu and sigma of target features with CLIP
            model_ppin = PPIN(f1.to(device),div=opts.div,ind=ind)
           
            model_ppin.to(device)

            optimizer = torch.optim.SGD(params=[
                {'params': model_ppin.parameters(), 'lr': opts.lr},
            ], lr= opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
            
            loss1 = 0
            while cur_itrs<opts.total_it:
                
                cur_itrs += 1
                if cur_itrs %100==0:
                    print(cur_itrs)

                optimizer.zero_grad()

                patches_low_hal_ = model_ppin() # (len(ind),C,H/div,W/div)               
                patches_low_hal= t1(patches_low_hal_)
                
                #target_features (hallucinated)
                target_features_from_low = model.backbone(patches_low_hal.to(device),trunc0=False,trunc1=True,trunc2=False,
                trunc3=False,trunc4=False,get0=False,get1=False,get2=False,get3=False,get4=False)
                target_features_from_low /= target_features_from_low.norm(dim=-1, keepdim=True).clone().detach()

                loss = (1- torch.cosine_similarity(text_target, target_features_from_low, dim=1)).mean()


                writer.add_scalar("loss"+str(i),loss,cur_itrs)

                loss.backward(retain_graph=False)
                
                optimizer.step()

            cur_itrs = 0

            for name, param in model_ppin.named_parameters():

                if param.requires_grad and name == 'style_mean':
                    learnt_mu_f1 = param.data  #(len(ind),C,1,1)
                elif param.requires_grad and name == 'style_std':
                    learnt_std_f1 = param.data  #(len(ind),C,1,1)
            
    
            for k in range(learnt_mu_f1.shape[0]):
                learnt_mu_f1_ = torch.from_numpy(learnt_mu_f1[k].detach().cpu().numpy())
                learnt_std_f1_ = torch.from_numpy(learnt_std_f1[k].detach().cpu().numpy())

                most = most_list[ind[k]]

                stats[str(most)+'_mu'].append(learnt_mu_f1_)
                stats[str(most)+'_std'].append(learnt_std_f1_)

            torch.cuda.empty_cache()
            f1.detach().cpu()
            images.detach().cpu()
            model_ppin.to('cpu')


    with open(opts.save_dir+'/saved_params.pkl', 'wb') as f:
        pickle.dump(stats, f)
            

main(random_styles=random_styles)
