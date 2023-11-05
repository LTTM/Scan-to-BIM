import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from shutil import rmtree

from torch import nn
import torch, os
torch.backends.cudnn.benchmark = True
from torch.optim import Adam
from torch.utils.data import DataLoader
import argparse

from model.cylinder3d.cylinder3d import Cylinder3D
from dataloaders.PCSdataset_cylinder import PCSDataset
from dataloaders.SemanticKITTIdataset import voxel_dataset
from util.lovasz_losses import lovasz_softmax
from util.metrics import Metrics
from util.common_util import log_pcs, schedule

#set seed for reproducibility
seed = 12345
np.random.seed(seed)
torch.manual_seed(seed)


def my_collate(batch):
    vox = [item[0] for item in batch]
    vox_lab = [item[1] for item in batch]
    data = [item[2] for item in batch]
    data_lab = [item[3] for item in batch]
    target = [item[4] for item in batch]
    return [vox, vox_lab, data, data_lab, target]


###### VALIDATION
def validate(writer, vset, vloader, epoch, model, device):  # PA, PP, mIoU
    metric = Metrics(vset.point_cloud_dataset.cnames, device=device)
    model.eval()
    with torch.no_grad():
        vbar = tqdm(vloader, "Validating Epoch %d" % (epoch + 1), total=len(vset))
        for i, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea) in enumerate(vbar):
            val_pt_fea_ten = [torch.from_numpy(j.detach().numpy()).type(torch.FloatTensor).to(device) for j in
                              val_pt_fea]
            val_grid_ten = [torch.from_numpy(j.detach().numpy()).to(device) for j in val_grid]
            y = val_vox_label.type(torch.LongTensor).to(device)
            o = model(val_pt_fea_ten, val_grid_ten)
            metric.add_sample(o.argmax(dim=1).flatten(), y.flatten())
    miou = metric.percent_mIoU()
    writer.add_scalar('mIoU', miou, epoch)
    print(metric)
    model.train()
    return miou, o, y


class Trainer():
    def __init__(self, args):

        lr0 = 1e-3
        lre = 1e-5
        eval_every_n_epochs = 10

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        test_name = "Cylinder_PCS" + args.test_name
        logdir = os.path.join("log/train", test_name)
        # rmtree(logdir, ignore_errors=True) #rimuove i files di log vecchi
        writer = SummaryWriter(logdir, flush_secs=.5)

        # Load model
        model = Cylinder3D(num_classes=args.num_classes + 1)
        if args.pretrain:
            new = model.state_dict()
            old = torch.load(args.pretrain)
            for k in new:
                if "out" not in k:
                    new[k] = old[k]
            model.load_state_dict(new)
            print("model restored from ", args.pretrain)
        model.to(device)
            
        # Load dataset
        dataset = PCSDataset
        dset = voxel_dataset(dataset(root_path="../PCSproject/Nuvole_di_punti", cube_edge=args.cube_edge),
                             grid_size=model.output_shape)
        dloader = DataLoader(dset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=2,
                             collate_fn=my_collate,
                             drop_last=True)

        vset = voxel_dataset(dataset(root_path="../PCSproject/Nuvole_di_punti", cube_edge=args.val_cube_edge,
                                     augment=False,
                                     split='val'), grid_size=model.output_shape)
        vloader = DataLoader(vset,
                             batch_size=args.val_batch_size,
                             # collate_fn=my_collate,
                             shuffle=False,
                             num_workers=2)

        loss = nn.CrossEntropyLoss(ignore_index=-1)

        # set up parameters for training
        steps_per_epoch = len(dset)//args.batch_size
        tot_steps = steps_per_epoch*args.epochs
        optim = Adam(model.parameters(), weight_decay=1e-2)

        ##########################################
        # TRAINING
        for e in range(args.epochs):
            torch.cuda.empty_cache()
            if e % eval_every_n_epochs == 0:
                if e >= 0:
                    miou, o, y = validate(writer, vset, vloader, e, model, device)
                    if miou > best_miou:
                        best_miou = miou
                        torch.save(model.state_dict(), logdir + "/val_best.pth")
                    #log_pcs(writer, dset, o, y)
                metrics = Metrics(dset.point_cloud_dataset.cnames, device=device)

            pbar = tqdm(dloader, total=steps_per_epoch,
                        desc="Epoch %d/%d, Loss: %.2f, mIoU: %.2f, Progress" % (e + 1, args.epochs, 0., 0.))

            for i, (_, train_vox_label, train_grid, _, train_pt_fea) in enumerate(pbar):
                step = i + steps_per_epoch * e
                lr = schedule(lr0, lre, step, tot_steps, .9)
                optim.param_groups[0]['lr'] = lr
                optim.zero_grad()

                train_pt_fea_ten = [torch.from_numpy(j).type(torch.FloatTensor).to(device) for j in train_pt_fea]
                train_grid_ten = [torch.from_numpy(j).to(device) for j in train_grid]
                y = torch.from_numpy(np.array(train_vox_label)).type(torch.LongTensor).to(device)

                o = model(train_pt_fea_ten, train_grid_ten)

                l = loss(o, y) #+ lovasz_softmax(torch.argmax(o), y)
                l.backward()

                p = o.detach().argmax(dim=1)
                metrics.add_sample(p.flatten(), y.flatten())

                optim.step()
                miou = metrics.percent_mIoU()
                pbar.set_description("Epoch %d/%d, Loss: %.2f, mIoU: %.2f, Progress" % (e + 1, args.epochs, l.item(), miou))

                writer.add_scalar('lr', lr, step)
                writer.add_scalar('loss', l.item(), step)
                writer.add_scalar('step_mIoU', miou, step)
                writer.add_scalars('IoU', {n: 100 * v for n, v in zip(metrics.name_classes, metrics.IoU())}, step)

            torch.save(model.state_dict(), logdir + "/latest.pth")

        ########################
        # VALIDATION
        miou = validate(writer, vset, vloader, e, model, device)
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), logdir + "/val_best.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25000, help='number of epochs to run')
    parser.add_argument("--batch_size", type=int, default=1, help='batch_size')
    parser.add_argument("--cube_edge", type=int, default=24, help='granularity of voxelization train')
    parser.add_argument("--val_cube_edge", type=int, default=256, default='granularity of voxelization val')
    parser.add_argument("--num_classes", type=int, default=8, help='number of classes to consider')
    parser.add_argument("--dset_path", type=str, default="/media/elena/M2SSD/PCSproject/Nuvole_di_punti", help='dataset path')
    parser.add_argument("--test_name", type=str, help='optional test name')
    parser.add_argument("--pretrain", type=str, help='pretrained model path')
    parser.add_argument("--loss", choices=['ce','cwce','ohem','mixed'], default='mixed', type=str, help='which loss to use')
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()

    trainer.writer.flush()
    trainer.writer.close()
