import numpy as np
from os import path
from scipy.spatial.transform import Rotation as R

import torch
from torch.utils.data import Dataset

class DICEADataset(Dataset):
    def __init__(self,
                 root_path="data/DICEA",
                 splits_path="data/DICEA",
                 split="train",
                 num_pts=122880,
                 augment=True,
                 repeat=1,
                 **kwargs):

        self.split = split
        self.root_path = root_path
        self.num_pts = num_pts
        self.augment = augment
        
        self.weights = self.init_weights()

        self.cmap = self.init_cmap()
        self.idmap = self.init_idmap()
        self.cnames = list(self.idmap.keys())

        self.items = repeat*[l.strip() for l in open(path.join(splits_path, split+'.txt'), 'r')]

    def init_cmap(self):
        cmap = np.array([[154, 205,  50], # beams
                         [169, 169, 169], # columns
                         [143,  48, 223], # doors
                         [255, 215,   0], # floors
                         [255, 255,   0], # roofs
                         [  0,   0, 255], # stairs
                         [255,   0,   0], # walls
                         [  0, 191, 255], # windows
                         [  0,   0,   0]], dtype=np.uint8) # unassigned
        return cmap
        
    def init_weights(self):
        pts = np.array( [512333, 421462, 93539, 147219, 891830, 1196754, 79483, 5458562, 108568] , dtype=np.int32)
        return 1/pts

    def init_idmap(self):
        idmap = {"Unassigned": 0,
                 "Beams":      1,
                 "Columns":    2,
                 "Doors":      3,
                 "Floors":     4,
                 "Roofs":      5,
                 "Stairs":     6,
                 "Walls":      7,
                 "Windows":    8} # i.e. empty space
        return idmap

    def __len__(self):
        return len(self.items)

    def color_label(self, lab, norm=True):
        if norm:
            return self.cmap[lab.numpy()]/255.
        else:
            return self.cmap[lab.numpy()]

    def clean_line(self, l):
        # -16.385012495881597,29.048298990139457,237.51045999999994,Walls,051771850
        x, y, z, l, _ = l.strip().split(',')
        return (float(x), float(y), float(z)), self.idmap[l]

    def __getitem__(self, item):
        fname = path.join(self.root_path, self.items[item])
        xyz, lab = zip(*[self.clean_line(l) for l in open(fname, 'r')])
        xyz, lab = np.array(xyz), np.array(lab)-1

        # center & rescale PC in [-1,1]
        xyz -= xyz.mean(axis=0)
        xyz /= np.linalg.norm(xyz, axis=1).max()

        if self.augment:

            per = np.random.permutation(xyz.shape[0])
            xyz, lab = xyz[per].copy(), lab[per].copy()

            if np.random.random()<.5:
                r = R.from_rotvec(np.pi*(np.random.random(3,)-.5)*np.array([0.1,0.1,1])).as_matrix()
                xyz = np.einsum('jk,nj->nk',r,xyz)

            # random subsampling
            size = self.num_pts
            cropto = np.random.randint(size//2, size)

            per = np.random.permutation(xyz.shape[0])[:cropto]
            xyz, lab = xyz[per].copy(), lab[per].copy()

        else:
            size = self.num_pts

        pad_to = size-xyz.shape[0]
        if pad_to>0:
            xyz = np.pad(xyz, ((0,pad_to),(0,0)))
            lab = np.pad(lab, ((0,pad_to)))

        # shift between [0,1]
        xyz = (xyz+1.)/2.

        return torch.from_numpy(xyz), torch.from_numpy(lab)
        #return torch.from_numpy(xyz).transpose(0,1).unsqueeze(-1), torch.from_numpy(lab).unsqueeze(-1)

    def to_plottable(self, x):
        return x.transpose(0,2)
