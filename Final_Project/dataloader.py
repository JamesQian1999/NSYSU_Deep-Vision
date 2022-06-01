"""Data utility functions."""
import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms


class SegmentationData(data.Dataset):

    def __init__(self, image_paths_file, test=0):
        self.root_dir_name = os.path.dirname(image_paths_file)
        self.test = test
        with open(image_paths_file) as f:
            self.image_names = f.read().splitlines()

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[ii] for ii in range(*key.indices(len(self)))]

        elif isinstance(key, int):
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            return self.get_item_from_index(key)
        else:
            raise TypeError("Invalid argument type.")

    def __len__(self):
        return len(self.image_names)

    def get_item_from_index(self, index):
        to_tensor = transforms.ToTensor()
        img_id = self.image_names[index].replace('.jpg', '')

        img = Image.open(os.path.join(self.root_dir_name,
                                      img_id + '.jpg')).convert('RGB')
        Resize = transforms.Resize((240, 240))

        h = np.shape(img)[0]
        w = np.shape(img)[1]

        img = Resize(img)
        img = to_tensor(img)

        target_labels = None
        if(self.test == 0):
            target = Image.open(os.path.join(self.root_dir_name,
                                             'annotations_instance',
                                             img_id + '.png'))
            target = Resize(target)
            target = np.array(target, dtype=np.int64)
            target_labels = target[..., 0]-1

            target_labels = torch.from_numpy(target_labels.copy())

        if(self.test == 0):
            return img, target_labels, h, w
        else:
            return img, img_id, h, w
