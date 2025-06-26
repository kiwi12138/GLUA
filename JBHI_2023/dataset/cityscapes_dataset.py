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

class cityscapesDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(256, 256), mean=0.5, scale=True, mirror=True, ignore_label=255, set='val'):
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
        self.set = set
        self.transform = get_transform()
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            # img_file = osp.join(self.root, "%s/%s" % (self.set, name))
            img_file = osp.join(self.root, "images/%s" % name)
            self.files.append({
                "img": img_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        name = datafiles["name"]

        # resize
        # image = image.resize(self.crop_size, Image.BILINEAR)
        image = self.transform(image)

        image = np.asarray(image, np.float32)

        size = image.shape
        # image = image[:, :, ::-1]  # change to BGR
        # image -= self.mean
        # image = image.transpose((2, 0, 1))

        return image.copy(), np.array(size), name

#
# if __name__ == '__main__':
#     dst = cityscapesDataSet("./data", is_transform=True)
#     trainloader = data.DataLoader(dst, batch_size=4)
