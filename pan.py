# -*- coding:utf-8 -*-
# @FileName  :test.py
# @Time      :2020/10/21 17:56
# @Author    :wangyan
import datetime
import cv2
import os
import sys
import numpy as np
import paddlex as pdx

for i in range(2179):

    model_fasterrcnn_path = "output/ppyolotiny/best_model"
    model_fasterrcnn = pdx.load_model(model_fasterrcnn_path)
    results = model_fasterrcnn.predict("test_pic/images_%d.jpg"%(i+1))  # faster-rcnn检测  83ms/pic
    #results = model_fasterrcnn.predict("test_pic/naiguo.jpg")
    pdx.det.visualize("test_pic/images_%d.jpg"%(i+1), results, threshold=0.5, save_dir='./output/testpic')
    #pdx.det.visualize("test_pic/naiguo.jpg", results, threshold=0.7, save_dir='./output')

print("End testing.")



