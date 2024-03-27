from utils import ext_transforms as et
from datasets import Cityscapes, gta5, BDD100K, Synthia, Mapillary


def get_dataset(dataset,data_root,crop_size,ACDC_sub="night",data_aug=True):
    """ Dataset And Augmentation
    """

    ###### Cityscapes
    if dataset == 'cityscapes':
        if data_aug:
            train_transform = et.ExtCompose([
                et.ExtRandomCrop(size=(crop_size, crop_size)),
                et.ExtColorJitter(brightness=0.8, contrast=0.8, saturation=0.8),
                et.ExtRandomHorizontalFlip(),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711]),
            ])
        else:
            train_transform = et.ExtCompose([
                et.ExtRandomCrop(size=(crop_size, crop_size)),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711]),
            ])

        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        train_dst = Cityscapes(root=data_root,dataset=dataset,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=data_root,dataset=dataset,
                             split='val', transform=val_transform)


    ####### ACDC
    if dataset == 'ACDC':
        train_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        train_dst = Cityscapes(root=data_root,dataset=dataset,
                               split='train', transform=train_transform, ACDC_sub = ACDC_sub)
        val_dst = Cityscapes(root=data_root,dataset=dataset,
                             split='val', transform=val_transform, ACDC_sub = ACDC_sub)


    ###### GTA5
    if dataset == "gta5":
        
        if data_aug:
            train_transform = et.ExtCompose([
                et.ExtRandomCrop(size=(crop_size, crop_size)),
                et.ExtColorJitter(brightness=0.8, contrast=0.8, saturation=0.8),
                et.ExtRandomHorizontalFlip(),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711]),
            ])
        else:
            train_transform = et.ExtCompose([
                et.ExtRandomCrop(size=(crop_size, crop_size)),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711]),
            ])

        val_transform = et.ExtCompose([
            et.ExtCenterCrop(size=(1046, 1914)),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        train_dst = gta5.GTA5DataSet(data_root, 'datasets/gta5_list/gtav_split_train.txt',transform=train_transform)
        val_dst = gta5.GTA5DataSet(data_root, 'datasets/gta5_list/gtav_split_val.txt',transform=val_transform)


    ###### Synthia
    if dataset == "synthia":
       
        if data_aug:
            train_transform = et.ExtCompose([
                et.RandomSizeAndCrop(size=(720,720)),
                et.ExtResize(size=(768,768)),
                et.ExtColorJitter(brightness=0.8, contrast=0.8, saturation=0.8),
                et.ExtRandomHorizontalFlip(),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711]),
            ])
        else:
            train_transform = et.ExtCompose([
                et.ExtRandomCrop(size=(768, 768)),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711]),
            ])
        
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711]),
        ])


        train_dst = Synthia(root=data_root, mode='train', transform=train_transform)
        val_dst = Synthia(root=data_root, mode='val', transform=val_transform)

    ###### BDD100k

    if dataset == 'bdd100k':
       
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        train_dst = []
        val_dst = BDD100K(root=data_root, mode='val', transform=val_transform)


    ###### Mapillary
    if dataset == 'mapillary':
        train_transform = et.ExtCompose([
            et.ResizeHeight(1536),
            et.CenterCropPad(1536),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        val_transform = et.ExtCompose([
            et.ResizeHeight(1536),
            et.CenterCropPad(1536),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        train_dst = []
        val_dst = Mapillary(root=data_root, quality='semantic', mode='val', transform=val_transform, test=False)

    return train_dst, val_dst