"""
Synthia Dataset Loader
"""
import logging
import os
import numpy as np
from PIL import Image
import imageio
from torch.utils import data


num_classes = 19
ignore_label = 255
img_postfix = '.png'


trainid_to_trainid = {
        0: ignore_label,  # void
        1: 10,            # sky
        2: 2,             # building
        3: 0,             # road
        4: 1,             # sidewalk
        5: 4,             # fence
        6: 8,             # vegetation
        7: 5,             # pole
        8: 13,            # car
        9: 7,             # traffic sign
        10: 11,           # pedestrian - person
        11: 18,           # bicycle
        12: 17,           # motorcycle
        13: ignore_label, # parking-slot
        14: ignore_label, # road-work
        15: 6,            # traffic light
        16: 9,            # terrain
        17: 12,           # rider
        18: 14,           # truck
        19: 15,           # bus
        20: 16,           # train
        21: 3,            # wall
        22: ignore_label  # Lanemarking
        }


def add_items(items, img_path, mask_path, mask_postfix, mode):
    """

    Add More items ot the list from the augmented dataset
    """

    if mode == "train":
        img_path = os.path.join(img_path, 'train')
        mask_path = os.path.join(mask_path, 'train')
    elif mode == "val":
        img_path = os.path.join(img_path, 'val')
        mask_path = os.path.join(mask_path, 'val')

    list_items = [name.split(img_postfix)[0] for name in
                os.listdir(img_path)]
    for it in list_items:
        item = (os.path.join(img_path, it + img_postfix),
                os.path.join(mask_path, it + mask_postfix))

        items.append(item)


def make_dataset(root, mode):
    """
    Assemble list of images + mask files

    fine -   modes: train/val/test/trainval    cv:0,1,2
    coarse - modes: train/val                  cv:na

    path examples:
    leftImg8bit_trainextra/leftImg8bit/train_extra/augsburg
    gtCoarse/gtCoarse/train_extra/augsburg
    """
    items = []

    assert mode in ['train', 'val', 'test', 'trainval']
    img_dir_name = 'RGB'
    img_path = os.path.join(root, img_dir_name)
    mask_path = os.path.join(root, 'GT', 'LABELS')
    mask_postfix = '.png'
    if mode == 'trainval':
        modes = ['train', 'val']
    else:
        modes = [mode]
    for mode in modes:
        logging.info('{} fine cities: '.format(mode))
        add_items(items, img_path, mask_path,
                    mask_postfix, mode)

    logging.info('Synthia-{}: {} images'.format(mode, len(items)))
    return items


class Synthia(data.Dataset):

    def __init__(self, root, mode, transform=None):
        self.mode = mode
        self.transform = transform


        self.imgs = make_dataset(root, mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

    def __getitem__(self, index):

        img_path, mask_path = self.imgs[index]

        img, mask = Image.open(img_path).convert('RGB'), imageio.imread(mask_path, format='PNG-FI')

        # This mask has pixel classes and instance IDs
        mask = np.array(mask, dtype=np.uint8)[:,:,0]
        mask_copy = mask.copy()
        for k, v in trainid_to_trainid.items():
            mask_copy[mask == k] = v


        mask = Image.fromarray(mask_copy.astype(np.uint8))

        # Transformations
    
        if self.transform is not None:
            img, mask = self.transform(img,mask)

        return img, mask

    def __len__(self):
        return len(self.imgs)