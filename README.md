# Fully Automated Scan-to-BIM via Point Cloud Instance Segmentation

Devid Campagnolo*, Elena Camuffo*, Umberto Michieli, Paolo Borin, Simone Milani and Andrea Giordano, _In Proceedings of the International Conference on Image Processing (ICIP) 2023_. 

 ![Static Badge](https://img.shields.io/badge/dataset-brightgreen?style=for-the-badge&labelColor=white&link=https%3A%2F%2Fdrive.google.com%2Fdrive%2Ffolders%2F1NmRegFS9HQQx7IJ7Klpn8mgWbW6bv9Eo%3Fusp%3Ddrive_link) 
![Static Badge](https://img.shields.io/badge/paper-lightblue?style=for-the-badge&labelColor=white&link=https%3A%2F%2Fieeexplore.ieee.org%2Fabstract%2Fdocument%2F10222064) 
![Static Badge](https://img.shields.io/badge/presentation-orange?style=for-the-badge&labelColor=white)
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />Our dataset is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

![image](https://github.com/LTTM/Scan-to-BIM/assets/63043735/d7bec320-4277-494c-8da1-5bfcf3469458)

### Abstract 

Digital Reconstruction through Building Information Models (BIM) is a valuable methodology for documenting and analyzing existing buildings. Its pipeline starts with geometric acquisition. (e.g., via photogrammetry or laser scanning) for accurate point cloud collection. However, the acquired data are noisy and unstructured, and the creation of a semantically-meaningful BIM representation requires a huge computational effort, as well as expensive and time-consuming human annotations. In this paper, we propose a fully automated scan-to-BIM pipeline. The approach relies on: (i) our dataset (HePIC), acquired from two large buildings and annotated at a point-wise semantic level based on existent BIM models; (ii) a novel ad hoc deep network (BIM-Net++) for semantic segmentation, whose output is then processed to extract instance information necessary to recreate BIM objects; (iii) novel model pretraining and class re-weighting to eliminate the need for a large amount of labeled data and human intervention.

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
