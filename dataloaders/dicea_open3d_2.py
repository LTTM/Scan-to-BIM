from os import path

from open3d._ml3d.datasets.base_dataset import BaseDataset
import numpy as np
import torch

class DICEA(BaseDataset):
    def __init__(self, name="DICEA"):
        super().__init__(name=name)
        # read file lists.

    def get_split(self, split):
        return DICEATrain(self, split=split)

    def is_tested(self, attr):
        pass
        # checks whether attr['name'] is already tested.

    def save_test_result(self, results, attr):
        pass
        # save results['predict_labels'] to file.


class DICEATrain():
    def __init__(self, dataset, split='train'):
        self.split = split
        self.path_list = []
        # collect list of files relevant to split.

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        path = self.path_list[idx]
        points, features, labels = self.read_pc(path)
        return {'point': points, 'feat': features, 'label': labels}

    def get_attr(self, idx):
        path = self.path_list[idx]
        name = path.split('/')[-1]
        return {'name': name, 'path': path, 'split': self.split}

###############################################################################


    def color_label(self, lab, norm=True):
        if norm:
            return self.cmap[lab.numpy()] / 255.
        else:
            return self.cmap[lab.numpy()]

    def clean_line(self, l):
        # -16.385012495881597,29.048298990139457,237.51045999999994,Walls,051771850
        x, y, z, l, _ = l.strip().split(',')
        return (float(x), float(y), float(z)), self.idmap[l]

    def readpc(self, fname):
        xyz, lab = zip(*[self.clean_line(l) for l in open(fname, 'r')])
        xyz, lab = np.array(xyz), np.array(lab) - 1

        # center & rescale PC in [-1,1]
        xyz -= xyz.mean(axis=0)
        xyz /= np.linalg.norm(xyz, axis=1).max()

        size = self.num_pts

        pad_to = size - xyz.shape[0]
        if pad_to > 0:
            xyz = np.pad(xyz, ((0, pad_to), (0, 0)))
            lab = np.pad(lab, ((0, pad_to)))

        # shift between [0,1]
        xyz = (xyz + 1.) / 2.
        points, features, labels = xyz, xyz, lab

        return torch.from_numpy(xyz), torch.from_numpy(lab)