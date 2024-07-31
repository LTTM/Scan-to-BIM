import os, sys
import argparse
import shutil

def main(args):

    path = args.dset_path
    pcs = os.listdir(path)
    
    with open('train.txt', 'r') as file:
        train = file.read().split('\n')[:-1]
    
    with open('test.txt', 'r') as file:
        test = file.read().split('\n')[:-1]

    with open('val.txt', 'r') as file:
        val = file.read().split('\n')[:-1]

    folders = ['1_Eremitani', '2_Castello']

    for fold in folders:
        os.makedirs(path+'/'+fold+'/'+'train', exist_ok=True)
        os.makedirs(path+'/'+fold+'/'+'test', exist_ok=True)
        os.makedirs(path+'/'+fold+'/'+'val', exist_ok=True)

        for pcs in train:
            if fold in pcs:
                name = pcs.split('/')[-1]
                shutil.move(path+'/'+fold+'/'+name, path+'/'+pcs)
        
        for pcs in test:
            if fold in pcs:
                name = pcs.split('/')[-1]
                shutil.copy(path+'/'+fold+'/'+name, path+'/'+pcs)

        for pcs in val:
            if fold in pcs:
                name = pcs.split('/')[-1]
                shutil.move(path+'/'+fold+'/'+name, path+'/'+pcs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset_path", type=str, default="/home/elena/Documents/Hepic", help='dataset path')
    args = parser.parse_args()

    main(args)