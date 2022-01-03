import glob
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from utils import (CLAHE, AdaptiveThreshold, CvtColor, EqualHist, Gabor,
                   Resize, train_model)


class FingerDataset(Dataset):
    def __init__(self, is_val=False) -> None:
        super().__init__()
        self.is_val = is_val
        dirs = glob.glob("ROI_finger/*")
        self.paths = []
        for dir in dirs:
            png_paths = glob.glob("{}/*".format(dir))
            if is_val:
                self.paths.append(png_paths[0])
            else:
                self.paths.extend(png_paths[1:])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        compose = transforms.Compose([
            EqualHist(),
            CLAHE(clip_limit=10, tile_grid_size=(8, 8)),
            # AdaptiveThreshold(block_size=31, c=2),
            # Gabor(kernel_size=(9, 9),
            #       sigma=0.3,
            #       theta=np.pi / 2,
            #       lambd=50,
            #       gamma=1,
            #       psi=0),
            Resize((224, 224)),
            CvtColor(cv2.COLOR_GRAY2RGB),
            transforms.ToTensor()
        ])
        path = self.paths[index]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_tensor = compose(img)
        label = os.path.split(path)[0].split("\\")[1]
        label = torch.tensor(int(label))
        return img_tensor, label


batch_size = 8
model = models.vgg19(pretrained=True)
model.classifier[6] = nn.Linear(in_features=4096, out_features=10)

datasets = {'train': FingerDataset(), 'val': FingerDataset(is_val=True)}
dataloaders = {
    x: DataLoader(datasets[x], batch_size=batch_size, shuffle=False)
    for x in ['train', 'val']
}
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}


optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
# 学习率衰减
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model = train_model(model,
                    dataloaders,
                    dataset_sizes,
                    criterion,
                    optimizer_ft,
                    exp_lr_scheduler,
                    num_epochs=25)
torch.save(model.state_dict(), "finger.pt")
