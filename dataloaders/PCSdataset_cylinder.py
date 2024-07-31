import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.transform import Rotation as R
import torch
from torch.utils.data import Dataset
from os import path

from dataloaders.SemanticKITTIdataset import nb_process_label, polar2cat, cart2polar


class PCSDataset(Dataset):
    def __init__(self,
                 root_path="/home/elena/Documents/Hepic",
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
        x, y, z, l, _ = l.strip().split(',')[:5]
        return (float(x), float(y), float(z)), self.idmap[l]

    def __getitem__(self, item):
        fname = path.join(self.root_path, self.items[item])
        xyz, lab = zip(*[self.clean_line(l) for l in open(fname, 'r')])
        xyz, lab = np.array(xyz), np.array(lab)

        return torch.from_numpy(xyz).unsqueeze(axis=0), torch.from_numpy(lab)

        """
        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 90) - np.pi / 4
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]
        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]
        # convert coordinate into polar coordinates

        if self.transform:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            xyz[:, 0:3] += noise_translate

        xyz_pol = cart2polar(xyz)

        max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
        min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)
        max_bound = np.max(xyz_pol[:, 1:], axis=0)
        min_bound = np.min(xyz_pol[:, 1:], axis=0)
        max_bound = np.concatenate(([max_bound_r], max_bound))
        min_bound = np.concatenate(([min_bound_r], min_bound))
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)
        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range / (cur_grid_size - 1)

        if (intervals == 0).any(): print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        voxel_position = polar2cat(voxel_position)

        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
        data_tuple = (voxel_position, processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)
        return_fea = return_xyz

        if self.return_test:
            data_tuple += (grid_ind, labels, return_fea, item)
        else:
            data_tuple += (grid_ind, labels, return_fea)
        return data_tuple
        """

