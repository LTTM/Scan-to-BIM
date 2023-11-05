import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.transform import Rotation as R
import torch
from torch.utils.data import Dataset
from os import path
from plyfile import PlyData

class S3DISDataset(Dataset):
    def __init__(self,
                 root_path="data/S3DIS",
                 splits_path="data/S3DIS",
                 split="train",
                 cube_edge=128,
                 augment=True):

        self.root_path = root_path
        self.cube_edge = cube_edge
        self.augment = augment

        self.cmap = self.init_cmap()
        self.idmap = self.init_idmap()
        self.weights = self.init_weights()      # inverse point frequency
        self.cnames = list(self.idmap.keys())

        self.items = [l.strip() for l in open(path.join(splits_path, split+'.txt'), 'r')]

    def init_cmap(self):
        cmap = np.array(  [[128, 64,128], # road
                           [244, 35,232], # sidewalk
                           [ 70, 70, 70], # building
                           [102,102,156], # wall
                           [190,153,153], # fence
                           [153,153,153], # pole
                           [250,170, 30], # traffic light
                           [220,220,  0], # traffic sign
                           [107,142, 35], # vegetation
                           [152,251,152], # terrain
                           [ 70,130,180], # sky
                           [220, 20, 60], # person
                           [  0,  0,142], # car
                           [  0,  0,  0]], dtype=np.uint8) # unassigned
        return cmap

    def init_idmap(self):
        idmap = {0: 'unassigned',
                 1: 'ceiling',
                 2: 'floor',
                 3: 'wall',
                 4: 'beam',
                 5: 'column',
                 6: 'window',
                 7: 'door',
                 8: 'table',
                 9: 'chair',
                10: 'sofa',
                11: 'bookcase',
                12: 'board',
                13: 'clutter'}
        idmap = {v:k for k,v in idmap.items()}
        return idmap
        
    def init_weights(self):
        pts = np.array( [3370714, 2856755, 4919229, 318158, 375640, 478001, 974733, 650464, 791496, 88727, 1284130, 229758, 2272837] , dtype=np.int32)
        return 1/pts

    def __len__(self):
        return len(self.items)

    def color_label(self, lab, norm=True):
        if norm:
            return self.cmap[lab.numpy()]/255.
        else:
            return self.cmap[lab.numpy()]

    def __getitem__(self, item):
        fname = path.join(self.root_path, self.items[item])
        
        data = PlyData.read(fname)
        xyz = np.array([data['vertex']['x'], data['vertex']['y'], data['vertex']['z']]).T #np.array([[x,y,z] for x,y,z,_,_,_,_ in data['vertex']])
        lab = data['vertex'][['class']].astype(int)+1 #np.array([l for _,_,_,_,_,_,l in data['vertex']])

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
