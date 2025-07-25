import os
import os.path as osp
import numpy as np
import random
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
import torchvision.transforms as transforms

def get_transform(grayscale=True, convert=True):
    transform_list = []
    if grayscale:
         transform_list.append(transforms.Grayscale(3))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)

class GTA5DataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(256, 256), mean=0.5, scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.transform = get_transform()
        self.id_to_trainid = {204:4,153:3,102:2,51:1,0:0}

        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "images/%s" % name)
            name = name.replace('_inputA__real_B', 'gtsB_')
            name = name.replace('_inputA__fake_B', 'gtsA_')
            name = name.replace('_inputA__fake', 'gtsA_')
            name = name.replace('_inputA_', 'gtsA_')
            label_file = osp.join(self.root, "labels/%s" % name.replace('_inputB__fake_A','gtsB_'))
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        # image = image.rotate(180)
        image = self.transform(image)

        label = Image.open(datafiles["label"])
        # label = label.rotate(180)
        name = datafiles["name"]

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        size = image.shape
        return image.copy(), label_copy.copy(), np.array(size), name

