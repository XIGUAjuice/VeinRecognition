#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax
#%%
img = cv2.imread("finger/003_2/04.jpg", cv2.IMREAD_GRAYSCALE)


def rotate_image(image, center, angle):
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image,
                            rot_mat,
                            image.shape[1::-1],
                            flags=cv2.INTER_LINEAR)
    return result


#%%
def show(img):
    cv2.imshow("img", img)
    cv2.waitKey(0)


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
        image_center = tuple(np.array(mask.shape[1::-1]) / 2)
        mask = rotate_image(mask, image_center,
                            rotation)[self.upper:self.lower]

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


class PalmROIExtracter:
    def __init__(self, scale=1) -> None:
        self.scale = scale

    def getMask(self, img):
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    def getCircleInside(self, mask):
        dist_map = cv2.distanceTransform(mask, cv2.DIST_L2,
                                         cv2.DIST_MASK_PRECISE)
        _, radius, _, center = cv2.minMaxLoc(dist_map)
        return center, radius

    def cropWrist(self, mask, center, radius):
        mask[int(center[1] + radius):, :] = 0
        return mask

    def getRotationPalm(self, mask, center, radius, draw=False):

        thetas = np.arange(360) + 90
        r = 1.5 * radius
        x = center[0] + r * np.cos(thetas / 180 * np.pi)
        y = center[1] + r * np.sin(thetas / 180 * np.pi)
        x = x.astype(int)
        y = y.astype(int)
        intersects = mask[y, x]
        turning = np.where(intersects[:-1] != intersects[1:])[0]
        point_1 = (x[turning[4]], y[turning[4]])
        point_2 = (x[turning[5]], y[turning[5]])
        mid_point_x = (point_1[0] + point_2[0]) // 2
        mid_point_y = (point_1[1] + point_2[1]) // 2
        theta = np.arctan2(center[1] - mid_point_y, center[0] - mid_point_x)
        rotate = theta / 2 / np.pi * 360 - 90
        if not draw:
            return rotate
        else:
            return rotate, point_1, point_2, mid_point_x, mid_point_y, intersects

    def getSquareInside(self, center, radius, scale=None):
        if scale is None:
            scale = self.scale
        upper = int(center[1] - radius / np.sqrt(2) * scale)
        lower = int(center[1] + radius / np.sqrt(2) * scale)
        left = int(center[0] - radius / np.sqrt(2) * scale)
        right = int(center[0] + radius / np.sqrt(2) * scale)
        return left, upper, right, lower

    def transform(self, img):
        mask = self.getMask(img)
        center, radius = self.getCircleInside(mask)
        mask = self.cropWrist(mask, center, radius)
        rotate = self.getRotationPalm(mask, center, radius)
        img = rotate_image(img, center, rotate)
        left, upper, right, lower = self.getSquareInside(center, radius)
        return img[upper:lower, left:right]

    def draw(self, img):
        fig, axs = plt.subplots(2, 3, figsize=(30, 20))
        plt.setp(axs, xticks=[], yticks=[])
        fig.tight_layout()

        axs[0, 0].imshow(img, cmap="gray")

        mask = self.getMask(img)
        axs[0, 1].imshow(mask, cmap="gray")

        center, radius = self.getCircleInside(mask)
        mask = self.cropWrist(mask, center, radius)
        result = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        cv2.circle(result, tuple(center), int(radius), (0, 0, 255), 2,
                   cv2.LINE_8, 0)
        cv2.circle(result, tuple(center), int(1.5 * radius), (0, 255, 0), 2,
                   cv2.LINE_8, 0)
        rotate, point_1, point_2, mid_point_x, mid_point_y, intersects = self.getRotationPalm(
            mask, center, radius, draw=True)
        cv2.circle(result, point_1, 8, (255, 165, 0), -1)
        cv2.circle(result, point_2, 8, (255, 165, 0), -1)
        cv2.circle(result, (mid_point_x, mid_point_y), 8, (255, 0, 0), -1)
        cv2.line(result, center, (mid_point_x, mid_point_y), (0, 255, 255), 2)
        axs[0, 2].imshow(result)

        axs[1, 0].plot(np.arange(intersects.shape[0]), intersects)

        result = rotate_image(result, center, rotate)
        axs[1, 1].imshow(result)

        left, upper, right, lower = self.getSquareInside(center, radius)
        cv2.rectangle(result, (left, upper), (right, lower), (0, 128, 0), 2)
        left, upper, right, lower = self.getSquareInside(center,
                                                         radius,
                                                         scale=0.8)
        cv2.rectangle(result, (left, upper), (right, lower), (255, 0, 0), 2)
        axs[1, 2].imshow(result)

        plt.show()