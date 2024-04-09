import os

import numpy as np
import torch
import wandb

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def schedule(lr0, lre, step, steps, power):
    return (lr0-lre)*(1-min(step/steps, 1))**power + lre # learning rate decrease poly 0.9


def log_pcs(writer, dset, pts, o, y):
    p, y = o[0].detach().argmax(dim=0).cpu(), y[0].cpu()
    cy = dset.color_label(y, norm=False).reshape(-1, 3)
    cp = dset.color_label(p, norm=False).reshape(-1, 3)
    my = y.flatten()>0
    
    if my.float().sum()>0:
        writer.add_mesh("labels", vertices=pts[:, my], colors=np.expand_dims(cy[my], 0))#, global_step=e)
        writer.add_mesh("preds", vertices=pts[:, my], colors=np.expand_dims(cp[my], 0))#, global_step=e)


def log_pcs_wandb(writer, dset, o, y):
    my_pred = o.detach().cpu().numpy()
    my_pred = np.argmax(my_pred, axis=1).flatten()
    my_gt = y.detach().cpu().numpy().flatten()
    my_pc = dset[0][0].reshape(3, my_gt.shape[0])
    cy = dset.point_cloud_dataset.color_label(my_gt, norm=False).reshape(-1, 3)[:, :3]
    cp = dset.point_cloud_dataset.color_label(my_pred, norm=False).reshape(-1, 3)[:, :3]
    my = y.flatten() > 0
    gt = np.vstack([my_pc, my_gt.reshape(1, len(my_gt))])
    p = np.vstack([my_pc, my_pred.reshape(1, len(my_pred))])

    # cosi logga i voxels
    wandb.log({"point_cloud_gt": wandb.Object3D(gt.T)})
    wandb.log({"point_cloud_pred": wandb.Object3D(p.T)})