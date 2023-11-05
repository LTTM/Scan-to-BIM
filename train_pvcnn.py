import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from shutil import rmtree

import torch, argparse
torch.backends.cudnn.benchmark = True
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch import nn

from open3d.ml.torch.models import PVCNN
from dataloaders.dicea_open3d import DICEADatasetOpen3D
from util.metrics import Metrics
from util.common_util import schedule, log_pcs

#set seed for reproducibility
seed = 12345
np.random.seed(seed)
torch.manual_seed(seed)


def conv_to_tensor(inputs):
    data = {}
    for k,v in inputs.items():
        if k == "coords" or k == "neighbor_indices" or k == "sub_idx" or  k == "interp_idx":
            new = [torch.from_numpy(t) for t in v]
            data = {**data,**{k:new}}
        else:
            data = {**data,**{k:torch.from_numpy(v)}}
    return data


###### VALIDATION
def validate(writer, vset, vloader, epoch, model, device): #PA, PP, mIoU
    metric = Metrics(vset.cnames[1:], device=device)
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(tqdm(vloader, "Validating Epoch %d" % (epoch + 1), total=len(vset))):
            o = model(data)
            y = data["label"].cuda()
            metric.add_sample(o.argmax(dim=2).detach().flatten(), y.flatten())
    miou = metric.percent_mIoU()
    acc = metric.percent_acc()
    prec = metric.percent_prec()
    writer.add_scalar('mIoU', miou, epoch)
    writer.add_scalar('PP', prec, epoch)
    writer.add_scalar('PA', acc, epoch)
    writer.add_scalars('IoU', {n:100*v for n,v in zip(metric.name_classes, metric.IoU()) if not torch.isnan(v)}, epoch)
    print(metric)
    model.train()
    return miou, o.swapaxes(1,2), y

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5000, help='number of epochs to run')
    parser.add_argument("--batch_size", type=int, default=8, help='batch_size')
    parser.add_argument("--cube_edge", type=int, default=96, help='granularity of voxelization train')
    parser.add_argument("--val_cube_edge", type=int, default=96, help='granularity of voxelization val')
    parser.add_argument("--num_classes", type=int, default=8, help='number of classes to consider')
    parser.add_argument("--dset_path", type=str, default="/media/elena/M2SSD/PCSproject/Nuvole_di_punti", help='dataset path')
    parser.add_argument("--test_name", type=str, help='optional test name')
    parser.add_argument("--pretrain", type=str, help='pretrained model path')
    parser.add_argument("--loss", choices=['ce','cwce','ohem','mixed'], default='mixed', type=str, help='which loss to use')
    args = parser.parse_args()
    
    lr0 = 2.5e-4
    lre = 1e-5
    eval_every_n_epochs = 10

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logdir = "log/train_pcs" + "_" + args.pretrain + args.test_name
    rmtree(logdir, ignore_errors=True)
    writer = SummaryWriter(logdir, flush_secs=.5)

    # Load model
    model = PVCNN(num_classes=args.num_classes) 
    if args.pretrain:
        new = model.state_dict()
        old = torch.load(args.pretrain)
        for k in new:
            if "out" not in k:
                new[k] = old[k]
        model.load_state_dict(new)
        print("model restored from ", args.pretrain)
    model.to('cuda')

    #Load dataset
    dataset = DICEADatasetOpen3D       
    dset = dataset(root_path=args.dset_path,
                   model=PVCNN,
                   cube_edge=args.cube_edge)
    dloader = DataLoader(dset,
                         batch_size=args.batch_size,
                         shuffle=True,
                         num_workers=4,
                         drop_last=True)

    vset = dataset(root_path=args.dset_path,
                   cube_edge=args.val_cube_edge,
                   model=PVCNN,
                   #augment=False,
                   split='val')
    vloader = DataLoader(vset,
                         batch_size=1,
                         shuffle=False,
                         num_workers=4)

    # set up parameters for training
    steps_per_epoch = len(dset)//args.batch_size
    tot_steps = steps_per_epoch*args.epochs
    optim = Adam(model.parameters(), weight_decay=1e-5)
    

    # to visualize point cloud
    pts = 2*torch.from_numpy(np.indices((args.val_cube_edge, args.val_cube_edge, args.val_cube_edge))
                             .reshape(3, -1).T).unsqueeze(0)/args.cube_edge - 1.
    best_miou = 0

    for e in range(args.epochs):
        torch.cuda.empty_cache()

        # Validation every n epochs
        if e % args.eval_every_n_epochs == 0:
            if e>=0:
                miou, o, y = validate(writer, vset, vloader, e, model, device)
                if miou>best_miou:
                    best_miou = miou
                    torch.save(model.state_dict(), logdir+"/val_best.pth")
                #log_pcs(writer, pts, o, y)
            metrics = Metrics(dset.cnames[1:], device=device)
        
        pbar = tqdm(dloader, total=steps_per_epoch, desc="Epoch %d/%d, Loss: %.2f, mIoU: %.2f, Progress"%(e+1, epochs, 0., 0.))

        for i, data in enumerate(pbar):

            step = i+steps_per_epoch*e

            lam = schedule(10, 1, step, tot_steps//2, .9)
            loss = nn.CrossEntropyLoss(ignore_index=-1)

            lr = schedule(lr0, lre, step, tot_steps, .9)
            optim.param_groups[0]['lr'] = lr
            
            optim.zero_grad()
            
            o = model(data)
            y = data["label"].cuda()
            l = loss(o.swapaxes(1,2), y.long()) 
            l.backward()

            metrics.add_sample(o.detach().argmax(dim=2).flatten(), y.flatten())

            optim.step()
            miou = metrics.percent_mIoU()
            pbar.set_description("Epoch %d/%d, Loss: %.2f, mIoU: %.2f, Progress"%(e+1, args.epochs, l.item(), miou))
            
            writer.add_scalar('lr', lr, step)
            writer.add_scalar('loss', l.item(), step)
            writer.add_scalar('step_mIoU', miou, step)

        torch.save(model.state_dict(), logdir+"/latest.pth")
        
    
    # EVALUATION
    miou, o, y = validate(writer, vset, vloader, e, model, device)
    if miou>best_miou:
        best_miou = miou
        torch.save(model.state_dict(), logdir+"/val_best.pth")
