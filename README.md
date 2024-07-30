# Fully Automated Scan-to-BIM via Point Cloud Instance Segmentation

Devid Campagnolo*, Elena Camuffo*, Umberto Michieli, Paolo Borin, Simone Milani and Andrea Giordano, _In Proceedings of the International Conference on Image Processing (ICIP) 2023_. 

[![Static Badge](https://img.shields.io/badge/dataset-brightgreen?style=for-the-badge&labelColor=white)](https://drive.google.com/drive/u/3/folders/1NmRegFS9HQQx7IJ7Klpn8mgWbW6bv9Eo)
[![Static Badge](https://img.shields.io/badge/paper-lightblue?style=for-the-badge&labelColor=white)](https://ieeexplore.ieee.org/abstract/document/10222064)
[![Static Badge](https://img.shields.io/badge/presentation-orange?style=for-the-badge&labelColor=white)](https://github.com/LTTM/Scan-to-BIM/files/12861649/Fully-Automated.Scan-to-BIM.via.Point.Cloud.Instance.Segmentation.Base.pdf)
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />Our dataset and codebase are licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

![image](https://github.com/LTTM/Scan-to-BIM/assets/63043735/d7bec320-4277-494c-8da1-5bfcf3469458)

### 🔥 News
- **HePIC**🏛️ dataset uploaded.
- **BIM-Net** codebase added - 5 models $\times$ 3 datasets.
- *Pretrained models --> coming soon!*

### Abstract 

Digital Reconstruction through Building Information Models (BIM) is a valuable methodology for documenting and analyzing existing buildings. Its pipeline starts with geometric acquisition. (e.g., via photogrammetry or laser scanning) for accurate point cloud collection. However, the acquired data are noisy and unstructured, and the creation of a semantically-meaningful BIM representation requires a huge computational effort, as well as expensive and time-consuming human annotations. In this paper, we propose a fully automated scan-to-BIM pipeline. The approach relies on: (i) our dataset (HePIC), acquired from two large buildings and annotated at a point-wise semantic level based on existent BIM models; (ii) a novel ad hoc deep network (BIM-Net++) for semantic segmentation, whose output is then processed to extract instance information necessary to recreate BIM objects; (iii) novel model pretraining and class re-weighting to eliminate the need for a large amount of labeled data and human intervention.

### 👩‍🍳 Recipe to use our code

1) Download **HePIC**🏛️ dataset from [here](https://drive.google.com/drive/u/3/folders/1NmRegFS9HQQx7IJ7Klpn8mgWbW6bv9Eo) and split the data as in `data/HePIC`.


2) Set up the environment with:
 ```
conda env create -f scan2bim.yml
 ```

 3) Train 🚀**BIM-Net++** with **HePIC**🏛️:
  ```
python train_pcs.py --loss mixed --dset_path [folder/of/your/dataset]
 ```

We include also training on:
- **HePIC with other models**: SegCloud, Cylinder3D, RandLA-Net, PVCNN.
- **BIM-Net with other datasets**: Arch, S3DIS.


### 🔍 Evaluation
You can download our pretrained models from [here]().
Results are reported here below.

| Model |    Dataset    |  PA |  PP | mIoU | Pretrained Weights |
|-------|:-------------:|-----:|-----:|-----:|---:|
SegCloud | **HePIC**🏛️ | 17.6 | 24.7| 13.2 |
Cylinder3D | **HePIC**🏛️ | 21.0 | 23.2 | 14.2 |
RandLA-Net | **HePIC**🏛️ |  35.6 | 56.2 | 28.8 |
PVCNN | **HePIC**🏛️ | 43.3 | 48.1 | 34.9 |
**BIM-Net** |  Arch | 26.0 | 39.8 | 18.4 |
**BIM-Net** |  S3DIS | 71.7 | 76.5 | 59.5 |
**BIM-Net** | **HePIC**🏛️ | 47.1 | 58.9 | 40.6 |
🚀**BIM-Net++** | **HePIC**🏛️ | 59.1 | 53.0 | 43.7 |

### Citation
If you find our work useful for your research, please consider citing:

 ```
@inproceedings{Campagnolo2023fully,
  author={Campagnolo, D. and Camuffo, E. and Michieli, U. and Borin, P. and Milani, S. and Giordano, A.},
  booktitle={IEEE International Conference on Image Processing (ICIP)}, 
  title={Fully Automated Scan-to-BIM Via Point Cloud Instance Segmentation}, 
  year={2023},
  pages={291-295},
  doi={10.1109/ICIP49359.2023.10222064}
  }
 ```
