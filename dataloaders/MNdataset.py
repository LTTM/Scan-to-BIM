import numpy as np
import os
from os import path

import torch
from torch.utils.data import Dataset

class MNDataset(Dataset):
    def __init__(self,
                 root_path="data/ModelNet40",
                 splits_path="data/ModelNet40",
                 split="train",
                 cube_edge=128,
                 max_d=15,
                 max_depth=2,
                 augment=False):

        self.root_path = root_path
        self.cube_edge = cube_edge
        self.max_d = max_d
        self.max_depth = max_depth
        self.augment = augment

        self.idmap = self.init_idmap()
        self.cnames = list(self.idmap.keys())

        self.split = split
        self.items = [l.strip().split(" ") for l in open(path.join(splits_path, split+'.txt'), 'r')]

    @staticmethod
    def read_off(fname):
        with open(fname, "r") as file:
            l0 = file.readline().strip()
            if 'OFF' != l0:
                l1 = l0.split("OFF")[1].split(' ')
            else:
                l1 = file.readline().strip().split(' ')
            n_verts, n_faces, n_dontknow = tuple([int(s) for s in l1])
            verts = np.array([[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)])
            faces = np.array([[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)])
            return verts, faces

    def init_idmap(self):
        idmap = {0: 'airplane',
                 1: 'bathtub',
                 2: 'bed',
                 3: 'bench',
                 4: 'bookshelf',
                 5: 'bottle',
                 6: 'bowl',
                 7: 'car',
                 8: 'chair',
                 9: 'cone',
                10: 'cup',
                11: 'curtain',
                12: 'desk',
                13: 'door',
                14: 'dresser',
                15: 'flower pot',
                16: 'glass box',
                17: 'guitar',
                18: 'keyboard',
                19: 'lamp',
                20: 'laptop',
                21: 'mantel',
                22: 'monitor',
                23: 'night stand',
                24: 'person',
                25: 'piano',
                26: 'plant',
                27: 'radio',
                28: 'range hood',
                29: 'sink',
                30: 'sofa',
                31: 'stairs',
                32: 'stool',
                33: 'table',
                34: 'tent',
                35: 'toilet',
                36: 'tv stand',
                37: 'vase',
                38: 'wardrobe',
                39: 'xbox'}
        idmap = {v:k for k,v in idmap.items()}
        return idmap

    def __len__(self):
        return len(self.items)

    def cast_face(self, vs, geom, depth=0, max_d=10):
        vs = vs.reshape(3,3)
        if np.all(vs==vs[0:1]):
            geom[tuple(vs[0:1].T)]=1
            return
        if np.all(np.abs(vs-vs[0:1])<2):
            geom[tuple(vs.T)]=1
            return

        mean = np.round(vs.mean(axis=0, keepdims=True)).astype(int)
        geom[tuple(mean.T)]=1

        if np.any(vs[0] != mean) and (depth < self.max_depth or np.any(np.abs(vs[0] - mean) > max_d)):
            vs_ = vs.copy()
            vs_[0] = mean
            self.cast_face(vs_.flatten(), geom, depth+1, max_d=self.max_d)

        if np.any(vs[1] != mean) and (depth < self.max_depth or np.any(np.abs(vs[1] - mean) > max_d)):
            vs_ = vs.copy()
            vs_[1] = mean
            self.cast_face(vs_.flatten(), geom, depth+1, max_d=self.max_d)

        if np.any(vs[2] != mean) and (depth < self.max_depth or np.any(np.abs(vs[2] - mean) > max_d)):
            vs_ = vs.copy()
            vs_[2] = mean
            self.cast_face(vs_.flatten(), geom, depth+1, max_d=self.max_d)

    def __getitem__(self, item):

        csplit = "train" if self.split in ["train", "val"] else "test"
        fname, lab = self.items[item]
        npy_fname = path.join(self.root_path,
                                fname.replace(
                                    csplit, self.split+"_"+str(self.cube_edge)).replace(
                                        'off', 'npy'))

        if path.exists(npy_fname):
            try:
                geom = np.load(npy_fname)
            except:
                raise ValueError("Broken NPY: "+npy_fname)
        else:
            fname = path.join(self.root_path, fname)

            verts, faces = self.read_off(fname)
            verts = verts-verts.min()
            verts = np.round((self.cube_edge-1)*(verts/verts.max())).astype(int)
            verts, idx = np.unique(verts, axis=0, return_inverse=True)
            faces = idx[faces]
            faces = np.unique(faces, axis=0)
            vfaces = verts[faces].reshape(-1, 9)

            geom = np.zeros((self.cube_edge, self.cube_edge, self.cube_edge), dtype=np.uint8)
            np.apply_along_axis(self.cast_face, 1, vfaces, geom, max_d=self.max_d)

            os.makedirs("/".join(npy_fname.split("/")[:-1]), exist_ok=True)
            np.save(npy_fname, geom)

        lab = int(lab)
        
        if self.augment:
            #flip x
            if np.random.random() < .5:
                geom = geom[::-1,...]
            #flip y
            if np.random.random() < .5:
                geom = geom[:,::-1,...]
            #flip z
            if np.random.random() < .5:
                geom = geom[...,::-1]
            #shift x
            if np.random.random() < .5:
                shift = np.random.randint(-self.cube_edge//4, self.cube_edge//4)
                geom = np.roll(geom, shift, 0)
                if shift >= 0:
                    geom[:shift,...] = 0
                else:
                    geom[shift:,...] = 0
            #shift y
            if np.random.random() < .5:
                shift = np.random.randint(-self.cube_edge//4, self.cube_edge//4)
                geom = np.roll(geom, shift, 1)
                if shift >= 0:
                    geom[:,:shift,...] = 0
                else:
                    geom[:,shift:,...] = 0
            #shift z
            if np.random.random() < .5:
                shift = np.random.randint(-self.cube_edge//4, self.cube_edge//4)
                geom = np.roll(geom, shift, 2)
                if shift >= 0:
                    geom[...,:shift] = 0
                else:
                    geom[...,shift:] = 0
        
        return torch.from_numpy(geom.astype(np.float32)).unsqueeze(0), lab
#python -c "from utils.MNdataset import MNDataset; d=MNDataset('K:/Uni/Datasets/ModelNet40','K:/Uni/Datasets/ModelNet40'); d[0]"
#for i, face in enumerate(faces):
#    vs = verts[face]
#    self.cast_face(vs, geom, max_d=self.max_d)
# idx = np.stack(np.where(geom>0)).T
# ptc = pv.PolyData(idx)
# ptc.plot()
