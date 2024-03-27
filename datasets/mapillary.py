"""
Mapillary Dataset Loader
"""

import os
import numpy as np
from PIL import Image
from torch.utils import data


# Convert this dataset to have labels from cityscapes
ignore_label = 255 #65
id_to_ignore_or_group = {}


def gen_id_to_ignore():
    global id_to_ignore_or_group
    for i in range(66):
        id_to_ignore_or_group[i] = ignore_label

    ### Convert each class to cityscapes one
    ### Road
    # Road
    id_to_ignore_or_group[13] = 0
    # Lane Marking - General
    id_to_ignore_or_group[24] = 0
    # Manhole
    id_to_ignore_or_group[41] = 0

    ### Sidewalk
    # Curb
    id_to_ignore_or_group[2] = 1
    # Sidewalk
    id_to_ignore_or_group[15] = 1

    ### Building
    # Building
    id_to_ignore_or_group[17] = 2

    ### Wall
    # Wall
    id_to_ignore_or_group[6] = 3

    ### Fence
    # Fence
    id_to_ignore_or_group[3] = 4

    ### Pole
    # Pole
    id_to_ignore_or_group[45] = 5
    # Utility Pole
    id_to_ignore_or_group[47] = 5

    ### Traffic Light
    # Traffic Light
    id_to_ignore_or_group[48] = 6

    ### Traffic Sign
    # Traffic Sign
    id_to_ignore_or_group[50] = 7

    ### Vegetation
    # Vegitation
    id_to_ignore_or_group[30] = 8

    ### Terrain
    # Terrain
    id_to_ignore_or_group[29] = 9

    ### Sky
    # Sky
    id_to_ignore_or_group[27] = 10

    ### Person
    # Person
    id_to_ignore_or_group[19] = 11

    ### Rider
    # Bicyclist
    id_to_ignore_or_group[20] = 12
    # Motorcyclist
    id_to_ignore_or_group[21] = 12
    # Other Rider
    id_to_ignore_or_group[22] = 12

    ### Car
    # Car
    id_to_ignore_or_group[55] = 13

    ### Truck
    # Truck
    id_to_ignore_or_group[61] = 14

    ### Bus
    # Bus
    id_to_ignore_or_group[54] = 15

    ### Train
    # On Rails
    id_to_ignore_or_group[58] = 16

    ### Motorcycle
    # Motorcycle
    id_to_ignore_or_group[57] = 17

    ### Bicycle
    # Bicycle
    id_to_ignore_or_group[52] = 18


def make_dataset(root, quality, mode):
    """
    Create File List
    """
    assert (quality == 'semantic' and mode in ['train', 'val'])
    img_dir_name = None
    if quality == 'semantic':
        if mode == 'train':
            img_dir_name = 'training'
        if mode == 'val':
            img_dir_name = 'validation'
        mask_path = os.path.join(root, img_dir_name,'labels')
    else:
        raise BaseException("Instance Segmentation Not support")

    img_path = os.path.join(root, img_dir_name, 'images')
    print(img_path)
    if quality != 'video':
        imgs = sorted([os.path.splitext(f)[0] for f in os.listdir(img_path)])
        msks = sorted([os.path.splitext(f)[0] for f in os.listdir(mask_path)])
        assert imgs == msks

    items = []
    c_items = os.listdir(img_path)
    if '.DS_Store' in c_items:
        c_items.remove('.DS_Store')

    for it in c_items:
        if quality == 'video':
            item = (os.path.join(img_path, it), os.path.join(img_path, it))
        else:
            item = (os.path.join(img_path, it),
                    os.path.join(mask_path, it.replace(".jpg", ".png")))
        items.append(item)
    return items


class Mapillary(data.Dataset):
    def __init__(self, root, quality, mode,
                 transform=None,
                 test=False):
        """
        class_uniform_tile_size = Class uniform tile size
        """
        gen_id_to_ignore()
        self.quality = quality
        self.mode = mode
        self.transform = transform

        # find all images
        self.imgs = make_dataset(root, quality, mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        if test:
            np.random.shuffle(self.imgs)
            self.imgs = self.imgs[:200]

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]

        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
        
        # img_name = os.path.splitext(os.path.basename(img_path))[0]

        mask = np.array(mask)
        mask_copy = mask.copy()
        for k, v in id_to_ignore_or_group.items():
            mask_copy[mask == k] = v
        mask = Image.fromarray(mask_copy.astype(np.uint8))

        # Image Transformations

        if self.transform is not None:
            img,mask = self.transform(img,mask)
     
        return img, mask

    def __len__(self):
        return len(self.imgs)