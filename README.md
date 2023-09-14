# Fully Automated Scan-to-BIM via Point Cloud Instance Segmentation

Devid Campagnolo*, Elena Camuffo*, Umberto Michieli, Paolo Borin, Simone Milani and Andrea Giordano, _In Proceedings of the International Conference on Image Processing (ICIP) 2023_.

[[DATASET](https://drive.google.com/drive/folders/1NmRegFS9HQQx7IJ7Klpn8mgWbW6bv9Eo?usp=drive_link)]
[[PAPER](https://ieeexplore.ieee.org/abstract/document/10222064)]

![image](https://github.com/LTTM/Scan-to-BIM/assets/63043735/73c3f25e-5415-4469-bf01-6b5ea31d3c72)
### Abstract 

Digital Reconstruction through Building Information Models (BIM) is a valuable methodology for documenting and analyzing existing buildings. Its pipeline starts with geometric
acquisition. (e.g., via photogrammetry or laser scanning) for accurate point cloud collection. However, the acquired data are noisy and unstructured, and the creation of a semantically-meaningful BIM representation requires a huge computational effort, as well as expensive and time-consuming human annotations. In this paper, we propose a fully automated scan-to-BIM pipeline. The approach relies on: (i) our dataset (HePIC), acquired from two large buildings and annotated at a point-wise semantic level based on existent BIM models; (ii) a novel ad hoc deep network (BIM-Net++) for semantic segmentation, whose output is then processed to extract instance information necessary to recreate BIM objects; (iii) novel model pretraining and class re-weighting to eliminate the need for a large amount of labeled data and human intervention.


