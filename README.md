# DS4002CS3
The purpose of this case study is to understand image recognition and object detection algorithms using the You Only Look Once (YOLO) model with images of a variety of fruits. 

## Software and Platform 
This project used Python programming through Google Colab on Mac platform but can also be reproduced using Jupyter Notebook on Windows Platform. 

Required installation: 
```
pip install ultralytics

import torch
import torchvision.transforms as transforms
import zipfile
import os
import pandas as pd
from PIL import Image
import numpy as np
import yaml
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
```
## Documentation
- LISCENCE.MD: proper citation for repository
- SCRIPTS folder: source code for project
- DATA folder: initial data (imported test/training images and fruit labels) and final cleaned data
- OUTPUT: figures and tables generated from scripts
- REFERENCES: sources used throughout project

## Reproducing Results
SCRIPTS: The script provided is the final script to clean the data, perform an EDA, training/testing YOLO model, and find precision of the model.

CLEAN DATA & EDA: read in original test/train data and labels (source: https://data.mendeley.com/datasets/5prc54r4rt), combine images and labels 
for ease of analysis. 
- OUTPUT: EDA graphs, merged_train_df, merged_test_df.

ANALYSIS: train YOLO model on images, and test model to detect certain fruits. Use mean average precision (mAP) to find precision of model.
- OUTPUT: mAP values

## References: 
1. World Health Organization. (n.d.). Hunger numbers stubbornly high for three consecutive years as global crises deepen: UN report. World Health Organization. https://www.who.int/news/item/24-07-2024-hunger-numbers-stubbornly-high-for-three-consecutive-years-as-global-crises-deepen--un-report 
2. Yolo algorithm for object detection explained [+examples]. YOLO Algorithm for Object Detection Explained [+Examples]. (n.d.). https://www.v7labs.com/blog/yolo-object-detection 
3. Latif, G. (2022, September 26). DeepFruits: Dataset of fruits images with different combinations for fruit recognition and calories estimation. Mendeley Data. https://data.mendeley.com/datasets/5prc54r4rt/1 
4. Latif, G., Mohammad, N., & Alghazo, J. (2023). DeepFruit: A dataset of fruit images for fruit classification and calories calculation. Data in Brief, 50, 109524. https://doi.org/10.1016/j.dib.2023.109524 

