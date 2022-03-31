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

'''绝对路径在工程稳定后需要改成相对路径'''
win_dir = "/home/hty/face_detection/dataset/"
lnx_dir = "/home/hty/face_detection/dataset/"
# work_dir = lnx_dir
work_dir = win_dir


if __name__ == "__main__":

    model_ppyolo_path = "output/ppyolotiny/best_model"
    model_ppyolo = pdx.load_model(model_ppyolo_path)

    testfile = "dataset/test_list.txt"
    with open(testfile, 'r') as f:
        for line in f.readlines():
            image_name = line.strip('\n')
            if os.path.isfile(image_name):
                starttime = datetime.datetime.now()
                results = model_ppyolo.predict(image_name)  # fasterrcnn检测  83ms/pic
                endtime = datetime.datetime.now()
                predict_time = (endtime - starttime).microseconds / 1000
                pdx.det.visualize(image_name, results, threshold=0.6, save_dir='./output/ppyolo_results')
                print(image_name + " runs : " + str(predict_time) + " ms")

    print("End testing.")
