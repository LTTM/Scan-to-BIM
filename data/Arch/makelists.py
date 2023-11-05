import numpy as np
from os import listdir, path

np.random.seed(12345)

dpath = "/media/elena/M2 SSD/Arch/Training"
tpath = "/media/elena/M2 SSD/Arch/Test"
fnames = [f for f in listdir(dpath) if path.isfile(path.join(dpath,f)) and not f.startswith('.')]
tnames = [f for f in listdir(tpath) if path.isfile(path.join(tpath,f)) and not f.startswith('.')]
per = np.random.permutation(len(fnames))
fnames = [fnames[i] for i in per]
per = np.random.permutation(len(tnames))
tnames = [tnames[i] for i in per]

with open("train.txt", "w") as f:
    for i in range(0, len(fnames)):
        f.write("Training/" + fnames[i]+"\n")

with open("val.txt", "w") as f:
    for i in range(len(tnames)):
        f.write("Test/" + tnames[i]+"\n")

with open("test.txt", "w") as f:
    for i in range(len(tnames)):
        f.write("Test/" + tnames[i]+"\n")