#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax
#%%
img = cv2.imread("finger/003_2/04.jpg", cv2.IMREAD_GRAYSCALE)


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image,
                            rot_mat,
                            image.shape[1::-1],
                            flags=cv2.INTER_LINEAR)
    return result


class FingerROIExtracter:
    def __init__(self, upper, lower) -> None:

        x = np.array([[-1, 0, 1]])
        self.kernel_left = np.repeat(x, 8, axis=0)
        x = np.array([[1, 0, -1]])
        self.kernel_right = np.repeat(x, 8, axis=0)
        self.upper = upper
        self.lower = lower

    def leftFill(self, left_row, left_edge):
        index = np.argmax(left_row)
        left_edge.append(index)
        left_row[:index] = 0
        left_row[index:] = 255

    def rightFill(self, right_row, right_edge):
        index = np.argmax(right_row)
        right_edge.append(index + 320)
        right_row[:index] = 255
        right_row[index:] = 0

    def getRotationFinger(self, left_edge, right_edge):
        left_edge = np.array(left_edge)
        right_edge = np.array(right_edge)
        center_line = ((left_edge + right_edge) // 2)
        x = np.arange(0, 350, 1)
        f = np.polyfit(x, center_line, 1)
        rotate = -np.arctan(f[0]) / 2 / np.pi * 360
        return rotate

    def transform(self, img):
        img = img[0:350]
        left = cv2.filter2D(img[:, :int(img.shape[1] / 2)], -1,
                            self.kernel_left)
        right = cv2.filter2D(img[:, int(img.shape[1] / 2):], -1,
                             self.kernel_right)

        left_edge = []
        right_edge = []
        np.apply_along_axis(self.leftFill, 1, left, left_edge=left_edge)
        np.apply_along_axis(self.rightFill, 1, right, right_edge=right_edge)
        mask = np.concatenate((left, right), axis=1)
        rotation = self.getRotationFinger(left_edge, right_edge)
        mask = rotate_image(mask, rotation)[self.upper:self.lower]

        all_white_column = np.all(mask, axis=0)
        margin = []
        for i in range(all_white_column.shape[0] - 1):
            if all_white_column[i] != all_white_column[i + 1]:
                margin.append(i)

        roi = img[self.upper:self.lower, margin[0] + 3:margin[1] - 3]
        return roi


# %%
class CLAHE:
    def __init__(self, clip_limit, tile_grid_size) -> None:
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(self.clip_limit, self.tile_grid_size)

    def transform(self, img):
        result = self.clahe.apply(img)
        return result


# %%
class AdaptiveThreshold:
    def __init__(self,
                 block_size,
                 c,
                 max_value=255,
                 adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                 threshold_type=cv2.THRESH_BINARY) -> None:
        self.max_value = max_value
        self.adaptive_method = adaptive_method
        self.threshold_type = threshold_type
        self.block_size = block_size
        self.c = c

    def transform(self, img):
        result = self.adaptive_threshold = cv2.adaptiveThreshold(
            img, self.max_value, self.adaptive_method, self.threshold_type,
            self.block_size, self.c)
        return result


#%%
class Gabor:
    def __init__(self, kernel_size, sigma, theta, lambd, gamma, psi) -> None:
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.theta = theta
        self.lambd = lambd
        self.gamma = gamma
        self.psi = psi
        self.kernel = cv2.getGaborKernel(self.kernel_size, self.sigma,
                                         self.theta, self.lambd, self.gamma,
                                         self.psi)
    
    def transform(self, img):
        result = cv2.filter2D(img, 1, self.kernel)
        return result
# %%
