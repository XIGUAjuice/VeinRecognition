#%%
import glob
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.core.fromnumeric import argmax
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import matplotlib.pyplot as plt
from utils import CLAHE, AdaptiveThreshold, CvtColor, EqualHist, Gabor, Resize


# %%
