from os import path
import numpy as np
import torch, yaml
from scipy.spatial.transform import Rotation as R
from open3d._ml3d.datasets.base_dataset import BaseDataset
from torch.utils.data import Dataset

#from model.randlanet import RandLANet
from open3d.ml.torch.models import PointTransformer

class DICEADatasetOpen3D(Dataset):
    name = "DICEA"
    with open('/home/elena/Documents/deeplearningproject/Open3D-ML/ml3d/configs/pointtransformer_dicea.yml') as f:
        cfg = yaml.safe_load(f)
    splits_path = "data/DICEA"

    def __init__(self,
                name="DICEA",
                root_path = "data/DICEA",
                splits_path = "data/DICEA",
                split = "train",
                num_pts = 122880,
                augment = True,
                repeat = 1,
                model = PointTransformer,
                **kwargs):
                super().__init__() #name=name)

                self.repeat = repeat
                self.splits_path = splits_path

                self.split = split
                self.root_path = root_path
                self.num_pts = num_pts
                self.augment = augment

                self.model = model
                self.weights = self.init_weights()
                self.cmap = self.init_cmap()
                self.idmap = self.init_idmap()
                self.cnames = list(self.idmap.keys())
                self.items = repeat * [l.strip() for l in open(path.join(splits_path, split + '.txt'), 'r')]
                self.path_list = self.items

    ############################
    # my functions
    def init_cmap(self):
        cmap = np.array([[154, 205, 50],  # beams
                         [169, 169, 169],  # columns
                         [143, 48, 223],  # doors
                         [255, 215, 0],  # floors
                         [255, 255, 0],  # roofs
                         [0, 0, 255],  # stairs
                         [255, 0, 0],  # walls
                         [0, 191, 255],  # windows
                         [0, 0, 0]], dtype=np.uint8)  # unassigned
        return cmap

    def init_weights(self):
        pts = np.array([512333, 421462, 93539, 147219, 891830, 1196754, 79483, 5458562, 108568], dtype=np.int32)
        return 1 / pts

    def init_idmap(self):
        idmap = {"Unassigned": 0,
                 "Beams": 1,
                 "Columns": 2,
                 "Doors": 3,
                 "Floors": 4,
                 "Roofs": 5,
                 "Stairs": 6,
                 "Walls": 7,
                 "Windows": 8}  # i.e. empty space
        return idmap

    def __getitem__(self, item):
        model = self.model(num_classes=len(self.idmap))
        data = self.getitem(item)
        data = {"point": data[0], "feat": data[0], "label": data[1]}
        if self.split == 'train':
            data = model.preprocess(data, self.get_attr(item))
            data = model.transform(data, self.get_attr(item))
            del data["search_tree"]
        else:
            model.inference_begin(data)
            data = model.inference_preprocess()
        return data

    def __len__(self):
        return len(self.items)

    def color_label(self, lab, norm=True):
        if norm:
            return self.cmap[lab.numpy()] / 255.
        else:
            return self.cmap[lab.numpy()]

    def clean_line(self, l):
        # -16.385012495881597,29.048298990139457,237.51045999999994,Walls,051771850
        x, y, z, l, _ = l.strip().split(',')
        return (float(x), float(y), float(z)), self.idmap[l]

    def getitem(self, item):
        fname = path.join(self.root_path, self.items[item])
        xyz, lab = zip(*[self.clean_line(l) for l in open(fname, 'r')])
        xyz, lab = np.array(xyz), np.array(lab) - 1

        # center & rescale PC in [-1,1]
        xyz -= xyz.mean(axis=0)
        xyz /= np.linalg.norm(xyz, axis=1).max()

        if self.augment:

            per = np.random.permutation(xyz.shape[0])
            xyz, lab = xyz[per].copy(), lab[per].copy()

            if np.random.random() < .5:
                r = R.from_rotvec(np.pi * (np.random.random(3, ) - .5) * np.array([0.1, 0.1, 1])).as_matrix()
                xyz = np.einsum('jk,nj->nk', r, xyz)

            # random subsampling
            size = self.num_pts
            cropto = np.random.randint(size // 2, size)

            per = np.random.permutation(xyz.shape[0])[:cropto]
            xyz, lab = xyz[per].copy(), lab[per].copy()

        else:
            size = self.num_pts

        pad_to = size - xyz.shape[0]
        if pad_to > 0:
            xyz = np.pad(xyz, ((0, pad_to), (0, 0)))
            lab = np.pad(lab, ((0, pad_to)))

        # shift between [0,1]
        xyz = (xyz + 1.) / 2.
        points, features, labels = xyz, xyz, lab

        return torch.from_numpy(xyz), torch.from_numpy(lab)
        # return torch.from_numpy(xyz).transpose(0,1).unsqueeze(-1), torch.from_numpy(lab).unsqueeze(-1)

    def to_plottable(self, x):
        return x.transpose(0, 2)

    ###################################
    # open3d dataset functions
    def get_split(self, split):
        self.split = split
        self.items = [l.strip() for l in open(path.join(self.splits_path, split + '.txt'), 'r')]
        return self

    def is_tested(self, attr):
        pass
        # checks whether attr['name'] is already tested.

    def save_test_result(self, results, attr):
        pass
        # save results['predict_labels'] to file.

    def get_data(self, idx):
        data = self.getitem(idx)
        points, features, labels = data[0], None, data[1]
        return {'point': points, 'feat': features, 'label': labels}

    def get_attr(self, idx):
        path = self.path_list[idx]
        name = path.split('/')[-1]
        return {'name': name, 'path': path, 'split': self.split}
