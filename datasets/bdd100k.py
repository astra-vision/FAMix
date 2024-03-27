"""
BDD100K Dataset Loader
"""
import logging
import os
import numpy as np
from PIL import Image
from torch.utils import data
import datasets.cityscapes_labels as cityscapes_labels


trainid_to_trainid = cityscapes_labels.trainId2trainId
img_postfix = '.jpg'


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
    img_dir_name = 'images'
    img_path = os.path.join(root, img_dir_name)
    mask_path = os.path.join(root, 'labels')
    mask_postfix = '.png'
    if mode == 'trainval':
        modes = ['train', 'val']
    else:
        modes = [mode]
    for mode in modes:
        logging.info('{} fine cities: '.format(mode))
        add_items(items, img_path, mask_path,
                    mask_postfix, mode)

    logging.info('BDD100K-{}: {} images'.format(mode, len(items)))
    return items


class BDD100K(data.Dataset):

    def __init__(self, root, mode,
                 transform=None):
        self.mode = mode
        self.transform = transform

        self.imgs  = make_dataset(root, mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

    def __getitem__(self, index):

        img_path, mask_path = self.imgs[index]

        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        # img_name = os.path.splitext(os.path.basename(img_path))[0]

        mask = np.array(mask)
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