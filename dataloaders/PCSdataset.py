import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.transform import Rotation as R
import torch
from torch.utils.data import Dataset
from os import path

class PCSDataset(Dataset):
    def __init__(self,
                 root_path="../../PCSproject/Nuvole_di_punti",
                 splits_path="data/HePIC",
                 fsl=None,
                 split="train",
                 cube_edge=128,
                 remapping=False,
                 pretrain=None,
                 augment=True):

        self.pretrain = pretrain
        self.remapping = remapping
        self.root_path = root_path
        self.cube_edge = cube_edge
        self.augment = augment

        self.weights = self.init_weights()

        self.idmap = self.init_idmap()
        if self.pretrain == "s3dis":
            pcs2s3dis = self.pcs2s3dis()
            self.idmap = {n:pcs2s3dis[self.idmap[n]] for n in self.idmap.keys() if self.idmap[n] == pcs2s3dis[self.idmap[n]]}

        self.cmap = self.init_cmap()
        if self.remapping:
            self.cmap = self.cmap[:3]

        if self.remapping:
            self.cnames = ["Unassigned", "Floors","Roofs","Walls"] #list(self.idmap.keys())
        else:
            self.cnames = list(self.idmap.keys())
        self.classmap = self.init_classmap()

        self.items = [l.strip() for l in open(path.join(splits_path, split+'.txt'), 'r')]

        if fsl and split=="train":
            self.items = self.items[:fsl]

    def init_weights(self):
        pts = np.array( [512333, 421462, 93539, 147219, 891830, 1196754, 79483, 5458562, 108568] , dtype=np.int32)
        return 1/pts

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
        
    def init_classmap(self):
        classmap = {0: 0,
                    1: 2,
                    2: 3,
                    3: 3,
                    4: 1,
                    5: 2,
                    6: 0,
                    7: 3,
                    8: 3} # i.e. empty space
        return classmap

    """
    def s3dis2pcs(self):
        idmap = {0:0, #'unassigned',
                 1:5, # 'ceiling',
                 2:4, # 'floor',
                 3:7, # 'wall',
                 4:1, # 'beam',
                 5:2, # 'column',
                 6:8, # 'window',
                 7:3, # 'door',
                 8:0, # 'table',
                 9:0, # 'chair',
                10:0, # 'sofa',
                11:0, # 'bookcase',
                12:0, # 'board',
                13:0} # 'clutter'}
        return idmap
    """

    def pcs2s3dis(self):
        idmap = {0:0, #'unassigned',
                 1:4, # 'beam',
                 2:5, # 'column',
                 3:7, # 'door',
                 4:2, # 'floor',
                 5:1, # 'ceiling',
                 6:8, # 'STAIRS',
                 7:3, # 'wall',
                 8:6} # 'window',
        return idmap

    def pcs2arch(self):
        idmap = {0:0, #'unassigned',
                 1:2, # 'beam',
                 2:1, # 'column',
                 3:5, #4, # 'door',
                 4:3, # 'floor',
                 5:8, # 'ceiling',
                 6:7, #6, # 'stairs',
                 7:6, #5, # 'wall',
                 8:4} # 'window',
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
        
        #mapping su 3 classi
        if self.remapping:
            for i in range(0,len(lab)):
                lab[i] = self.classmap[lab[i]]

        #mapping su s3dis
        if self.pretrain == "s3dis":
            map = self.pcs2s3dis()
            for i in range(0,len(lab)):
                lab[i] = map[lab[i]]
        #mapping su arch
        if self.pretrain == "arch":
            map = self.pcs2arch()
            for i in range(0,len(lab)):
                lab[i] = map[lab[i]]

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
