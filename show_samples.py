import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
from tqdm import tqdm

import torch
torch.backends.cudnn.benchmark = True
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from model.segcloud import ESegCloud
from dataloaders.PCSdataset import PCSDataset

if __name__ == '__main__':

    cube_edge = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = ESegCloud()
    model.load_state_dict(torch.load("log/train/latest.pth"))
    model.to('cuda')
    dset = PCSDataset(cube_edge=cube_edge,
                      augment=False,
                      split='val')

    ids = np.indices((cube_edge, cube_edge, cube_edge)).reshape(3, -1).T

    with torch.no_grad():
        for x, y in dset:
            y -= 1
            cy = dset.color_label(y).reshape(-1, 4)
            my = y.flatten()>=0
            x = x.to(device).unsqueeze(0)
            p = model(x).argmax(dim=1).squeeze(0).cpu()
            cp = dset.color_label(p).reshape(-1, 4)
            
            ypcd = o3d.geometry.PointCloud()
            ypcd.points = o3d.utility.Vector3dVector(ids[my])
            ypcd.colors = o3d.utility.Vector3dVector(cy[my,:3])
            ppcd = o3d.geometry.PointCloud()
            ppcd.points = o3d.utility.Vector3dVector(ids[my])
            ppcd.colors = o3d.utility.Vector3dVector(cp[my,:3])
            
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=1280, height=640)
            vis.add_geometry(ypcd)
            vis.run()
            
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=1280, height=640)
            vis.add_geometry(ppcd)
            vis.run()
            
            vis.destroy_window()