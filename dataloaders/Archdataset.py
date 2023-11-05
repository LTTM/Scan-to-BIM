import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.transform import Rotation as R
import torch
from torch.utils.data import Dataset
from os import path

class ArchDataset(Dataset):
    def __init__(self,
                 root_path="/media/elena/M2 SSD/Arch",
                 splits_path="data/Arch",
                 split="train",
                 cube_edge=128,
                 augment=True):

        self.root_path = root_path
        self.cube_edge = cube_edge
        self.augment = augment

        self.idmap = self.init_idmap()
        self.cmap = self.init_cmap()
        self.cnames = list(self.idmap.keys())

        self.items = [l.strip() for l in open(path.join(splits_path, split+'.txt'), 'r')]

    def init_cmap(self):
        cmap = np.array([[  0,   0, 250], # arch
                         [250,   0,   0], # column
                         [178,   0, 130], # moldings
                         [ 84,   0,   0], # floor
                         [250, 230,   0], # door_window
                         [210, 210, 155], # wall
                         [  0, 200,   0], # stairs
                         [250, 170,   0], # vault
                         [180,   0,   0], # roof
                         [  0, 230, 230],# other
                         [  0,   0,   0]], dtype=np.uint8) # unassigned
        return cmap

    def init_idmap(self):           # ATTENZIONE ALL'UNLABEL
        idmap = {"Unassigned": -1,
                 "Arch":        0,
                 "Column":      1,
                 "Moldings":    2,
                 "Floor":       3,
                 "Door_window": 4,
                 "Wall":        5,
                 "Stairs":      6,
                 "Vault":       7,
                 "Roof":        8,
                 "Other":       9}
        idmap = {k:v+1 for k,v in idmap.items()}
        return idmap

    def __len__(self):
        return len(self.items)

    def color_label(self, lab, norm=True):
        if norm:
            return self.cmap[lab.numpy()]/255.
        else:
            return self.cmap[lab.numpy()]

    def clean_line(self, l):
        # x, y, z, r, g, b, label, Nx, Ny, Nz
        x, y, z, _, _, _, lb, _, _, _ = l.strip().split(' ')
        return (float(x), float(y), float(z)), int(lb)+1

    def __getitem__(self, item):
        fname = path.join(self.root_path, self.items[item])
        #xyz, lab = zip(*[self.clean_line(l) for l in open(fname, 'r')])
        data = np.loadtxt(fname, usecols=(0, 1, 2, 6))
        xyz, lab = data[:, :3], data[:, 3]+1
        #xyz, lab = np.array(xyz), np.array(lab)

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
