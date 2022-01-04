import glob
import os

import cv2

from utils import FingerROIExtracter, PalmROIExtracter


def saveROI(paths, roi_extracter, dir):
    try:
        os.makedirs(dir)
    except Exception:
        print("文件夹已存在")
    i = 1
    for path in paths:
        try:
            img = cv2.imread(path, cv2.cv2.IMREAD_GRAYSCALE)
            roi = roi_extracter(img)
            cv2.imwrite("{}/{}.png".format(dir, i), roi)
            i = i + 1
        except Exception:
            print(path)


""" 生成ROI数据集 """
dirs = glob.glob("finger/*")
finger_roi_extracter = FingerROIExtracter()
i = 0
for dir in dirs:
    paths = glob.glob("{}/*.jpg".format(dir))
    saveROI(paths, finger_roi_extracter, "ROI_finger/{}".format(i))
    i = i + 1

#%%
dirs = glob.glob("palm/*")
palm_roi_extracter = PalmROIExtracter()
i = 0
for dir in dirs:
    paths = glob.glob("{}/*.bmp".format(dir))
    saveROI(paths, palm_roi_extracter, "ROI_palm/{}".format(i))
    i = i + 1
