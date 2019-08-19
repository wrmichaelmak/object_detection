# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 23:47:08 2019

@author: Michael Mak
"""

import cv2
import os

img_path = 'C:\\Users\\Michael Mak\\Desktop\Internship 2019\\Ongoing\\dataset\\training_set'
all_file = os.listdir(img_path)
os.chdir(img_path) # very important
list = ['106','110','137','150','153','165','166','180','19','190','196','204','208','229','25','28','535']
for i in range(len(list)):
    # print(filename)
    img=cv2.imread(list[i]+'.jpg')
    cv2.imwrite(list[i]+'.jpg',img,[int(cv2.IMWRITE_JPEG_QUALITY), 100])