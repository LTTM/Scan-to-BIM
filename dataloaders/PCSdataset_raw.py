import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.transform import Rotation as R
import torch
from torch.utils.data import Dataset
from os import path

class PCSDataset(Dataset):
    name = "PCS"
    
    def __init__(self,
                 root_path="data/PCS",
                 splits_path="data/PCS",
                 split="train",
                 cube_edge=128,
                 augment=True):
        
        self.root_path = root_path
        self.cube_edge = cube_edge
        self.augment = augment

        self.cmap = self.init_cmap()
        self.idmap = self.init_idmap()
        self.cnames = list(self.idmap.keys())

        self.items = [l.strip() for l in open(path.join(splits_path, split+'.txt'), 'r')]

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
        xyz, lab = np.array(xyz), np.array(lab)

        if self.augment and np.random.random()<.5:
            xyz += np.random.randn(*xyz.shape)

        # center & rescale PC in [-1,1]
        xyz -= xyz.mean(axis=0)
        xyz /= np.abs(xyz).max()

        if self.augment:

            # random rotation
            if np.random.random()<.5:
                r = R.from_rotvec(np.pi*(np.random.random(3,)-.5)*np.array([0.1,0.1,1])).as_matrix()
                xyz = np.einsum('jk,nj->nk',r,xyz)
                xyz /= np.abs(xyz).max()

            # random shift
            if np.random.random()<.5:
                xyz -= np.random.random((3,))*2-1.
            else:
                xyz += 1

            # random rescale & crop
            if np.random.random()<.5:
                if np.random.random()<.5:
                    xyz = np.round(xyz*(self.cube_edge//2)*np.random.random()).astype(int)
                else:
                    xyz = np.round(xyz*(self.cube_edge//2)/np.random.random()).astype(int)
            else:
                xyz = np.round(xyz*(self.cube_edge//2)).astype(int)

            valid = np.logical_and(np.all(xyz>-1, axis=1), np.all(xyz<self.cube_edge, axis=1))
        else:
            xyz += 1
            xyz = np.round(xyz*(self.cube_edge//2)).astype(int)
            valid = np.logical_and(np.all(xyz>-1, axis=1), np.all(xyz<self.cube_edge, axis=1))

        xyz = xyz[valid,:]
        lab = lab[valid]

        geom = np.zeros((self.cube_edge, self.cube_edge, self.cube_edge), dtype=np.float32)
        geom[tuple(xyz.T)] = 1

        labs = np.zeros((self.cube_edge, self.cube_edge, self.cube_edge), dtype=np.long)
        labs[tuple(xyz.T)] = lab

        return torch.from_numpy(geom).unsqueeze(0), torch.from_numpy(labs)
