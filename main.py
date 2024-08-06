from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
from torch.utils import data
from datasets import Cityscapes
from metrics import StreamSegMetrics
import torch
import torch.nn as nn
import pickle
from copy import deepcopy
from utils.get_dataset import get_dataset
from utils.freeze import * 
from torch.nn.functional import unfold


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options
    parser.add_argument("--dataset", type=str, default='gta5',
                        choices=['cityscapes','ACDC','gta5','synthia','mapillary','bdd100k'], help='Name of dataset')
    parser.add_argument("--data_root", type=str, default='/datasets_master/gta5',
                        help="path to Dataset")
    parser.add_argument("--ACDC_sub", type=str, default="night",
                        help = "specify which subset of ACDC  to use")

    # Backbone Options

    parser.add_argument("--BB", type = str, default = "RN50",
                        help = "backbone of the segmentation network")
    parser.add_argument("--OS", type= int, default=16,
                        help = "output stride")

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--total_itrs", type=int, default=40e3,
                        help="epoch number (default: 40k)")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="learning rate (default: 0.1)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help='batch size (default: 8)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=768)
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--val_interval", type=int, default=750,
                        help="iteration interval for eval (default: 750)")

    parser.add_argument("--ckpts_path", type = str ,
                        help="path for checkpoints saving")
    parser.add_argument("--data_aug", action='store_true', default=False)
    parser.add_argument("--num_classes", type=int, default=19,
                        help="number of classes to be considered for segmentation")
    parser.add_argument("--path_for_stats",type=str, help="path for the optimized stats")
    
    parser.add_argument("--transfer", action='store_true',default=True)
    return parser


def validate(model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()

    with torch.no_grad():

        for i, (images, labels) in tqdm(enumerate(loader), total=len(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs, _ = model(images)
            
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
           
            metrics.update(targets, preds)

        score = metrics.get_results()
    return score


def main():
    opts = get_argparser().parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    # INIT
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
  
    train_dst,val_dst = get_dataset(opts.dataset,opts.data_root,opts.crop_size,opts.ACDC_sub,
                                    data_aug=opts.data_aug)

    if not opts.test_only:
        train_loader = data.DataLoader(
            train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=4,
        drop_last=True)  # drop_last=True to ignore single-image batches.

    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=4)
    
    print("Dataset: %s, Train set: %d, Val set: %d" %
    (opts.dataset, len(train_dst), len(val_dst)))


    # Set up model
    model = network.modeling.__dict__['deeplabv3plus_resnet_clip'](num_classes=opts.num_classes, BB= opts.BB, OS=opts.OS)
    model.backbone.attnpool = nn.Identity()

    # freeze layers
    if opts.dataset == "gta5" or opts.dataset == "synthia":
        freeze_1_2_3_p4(model)
    elif opts.dataset == "cityscapes":
        freeze_1_2_3(model)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)
    
    # Set up optimizers
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
        ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean',label_smoothing=0.1)

    def save_ckpt(path,model,optimizer,scheduler,best_score):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)
    
    if not opts.test_only:
        utils.mkdir(opts.ckpts_path)
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        if opts.continue_training:
            checkpoint["optimizer_state"] = deepcopy(checkpoint["optimizer_state"])
            checkpoint["scheduler_state"] = deepcopy(checkpoint["scheduler_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" %opts.ckpt)
        del checkpoint  # free memory
        
    else:
        print("[!] Retrain")
        model.to(device)
        
    # ==========   Train Loop   ==========#


    #     return

    interval_loss = 0

    if not opts.test_only: 
        with open(opts.path_for_stats+'/saved_params.pkl', 'rb') as f:

            loaded_dict_patches = pickle.load(f)

        relu = nn.ReLU(inplace=True)


    if opts.test_only:
       
        model.eval()

        val_score = validate(model=model, loader=val_loader, device=device, metrics=metrics)

        print(metrics.to_str(val_score))
        print(val_score["Mean IoU"])
        print(val_score["Class IoU"])
        save_txt = "logs_PODA_val_gta5.txt"
        with open(save_txt, 'a') as f:
            f.write(f'{val_score["Mean IoU"]}\n')
        return


    while True:  # cur_itrs < opts.total_itrs:
    # =====  Train  =====

        if opts.dataset == "gta5" or opts.dataset == "synthia":
            model.backbone.layer4[2].train()
        elif opts.dataset == "cityscapes":
            model.backbone.layer4.train()
        model.classifier.train()

        cur_epochs += 1
       
        for i, (images, labels) in enumerate(train_loader):
            
            cur_itrs += 1
            images = images.to(device, dtype=torch.float32)
            if labels.type() != 'torch.FloatTensor':
                labels = labels.to(torch.float32)
            # labels = labels.to(device, dtype=torch.long)
           
            optimizer.zero_grad()
            
            labels_ = labels.unsqueeze(1)  # (B,1,768,768)
            # lbl_patches = divide_in_patches(labels_,3)

            lbl_patches = unfold(labels_, kernel_size=256, stride=256).permute(-1,0,1)
            lbl_patches = lbl_patches.reshape(lbl_patches.shape[0],lbl_patches.shape[1],1,256,256) #### (div*div, B, 1, H/div, W/div)
            lbl_patches = lbl_patches.to(torch.long)

            most_list = []
            for j in range(len(lbl_patches)): ### iterate on dim 0 (div*div)
                most = [Cityscapes.name(torch.mode(torch.flatten(lbl_patches[j][k])).values) if torch.mode(torch.flatten(lbl_patches[j][k])).values != 255 else 255 for k in range(lbl_patches[0].shape[0])]
                most_list.append(most) #len=div*div , each element list of B elements
            

            beta_dist = torch.distributions.beta.Beta(0.1, 0.1)
            s = beta_dist.sample((opts.batch_size, 256, 1, 1)).to('cuda')

            outputs,features = model(images, transfer=opts.transfer,mix=True,most_list=most_list,saved_params=loaded_dict_patches,activation=relu,s=s)
        
            ##############################################################################################################################################
            labels = labels.to(device, dtype=torch.long)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
          

            writer.add_scalar("loss",loss,cur_itrs)

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss_clip=%f" %
                    (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0 or cur_itrs == opts.total_itrs:
             
                save_ckpt(opts.ckpts_path+'/latest_%s_%s_'%
                        ('deeplabv3plus_resnet_clip', opts.dataset)+str(cur_itrs)+'.pth' ,model,optimizer,scheduler,best_score)
               
                print("validation...")
                
                model.eval()

                val_score = validate(model=model, loader=val_loader,device=device, metrics=metrics)
                
                print(metrics.to_str(val_score))
            
                
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt(opts.ckpts_path+'/best_%s_%s.pth' %
                            ('deeplabv3plus_resnet_clip', opts.dataset),model,optimizer,scheduler,best_score)

                writer.add_scalar("mIoU", val_score['Mean IoU'] ,cur_itrs)

                if opts.dataset == "gta5":
                    model.backbone.layer4[2].train()
                elif opts.dataset == "cityscapes":
                    model.backbone.layer4.train()
        
                model.classifier.train()


            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                return
            

if __name__ == '__main__':
    main()
