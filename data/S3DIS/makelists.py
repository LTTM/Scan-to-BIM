import numpy as np
from os import listdir, path

np.random.seed(12345)

dpath = "D:/Datasets/S3DIS"
fnames = [f for f in listdir(dpath) if path.isfile(path.join(dpath,f)) and not f.startswith('.')]
per = np.random.permutation(len(fnames))
fnames = [fnames[i] for i in per]

with open("train.txt", "w") as f:
    for i in range(0, len(fnames)-len(fnames)//5):
        f.write(fnames[i]+"\n")

with open("val.txt", "w") as f:
    for i in range(len(fnames)-len(fnames)//5, len(fnames)-len(fnames)//10):
        f.write(fnames[i]+"\n")

with open("test.txt", "w") as f:
    for i in range(len(fnames)-len(fnames)//10, len(fnames)):
        f.write(fnames[i]+"\n")