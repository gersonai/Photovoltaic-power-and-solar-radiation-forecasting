#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
from torchvision import transforms

class SunsetSunny(nn.Module):
    
    def __init__(self):
        
        super(SunsetSunny, self).__init__()

        self.convolNet = nn.Sequential(
            # Conv #1
            nn.Conv2d(
                in_channels=3,
                out_channels=24,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            
            # Conv #2 
            nn.Conv2d(
                in_channels=24,
                out_channels=48,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )
        
        self.fulcon = nn.Sequential(
            # Fully Connected Layer 1
            nn.Linear(
                in_features=12288,
                out_features=1024,
                bias=True
            ),
            nn.Dropout(p=0.4),
            nn.ReLU(),
            
            # Fully Connected Layer 2
            nn.Linear(
                in_features=1024,
                out_features=1024,
                bias=True
            ),
            nn.Dropout(p=0.4),
            nn.ReLU(),
            
            # Regression Layer
            nn.Linear(
                in_features=1024,
                out_features=1,
                bias=True
            )
        )

    def forward(self, batch):
        result = self.convolNet(batch)
        result = result.view(batch.size(0), -1) # reshap pour obtenir une matrice 2 Dims
        result = self.fulcon(result)
        return result



class SunsetCloudy(nn.Module):
    
    def __init__(self):
        
        super(SunsetCloudy, self).__init__()

        self.convolNet = nn.Sequential(
            # Conv #1
            nn.Conv2d(
                in_channels=3,
                out_channels=24,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            
            # Conv #2 
            nn.Conv2d(
                in_channels=24,
                out_channels=48,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )
        
        self.fulcon = nn.Sequential(
            # Fully Connected Layer 1
            nn.Linear(
                in_features=12288,
                out_features=1024,
                bias=True
            ),
            nn.Dropout(p=0.4),
            nn.ReLU(),
            
            # Fully Connected Layer 2
            nn.Linear(
                in_features=1024,
                out_features=1024,
                bias=True
            ),
            nn.Dropout(p=0.4),
            nn.ReLU(),
            
            # Regression Layer
            nn.Linear(
                in_features=1024,
                out_features=1,
                bias=True
            )
        )

    def forward(self, batch):
        result = self.convolNet(batch)
        result = result.view(batch.size(0), -1) # reshap pour obtenir une matrice 2 Dims
        result = self.fulcon(result)
        return result
    

    
    
class SunsetOvercast(nn.Module):
    
    def __init__(self):
        
        super(SunsetOvercast, self).__init__()

        self.convolNet = nn.Sequential(
            # Conv #1
            nn.Conv2d(
                in_channels=3,
                out_channels=24,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            
            # Conv #2 
            nn.Conv2d(
                in_channels=24,
                out_channels=48,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )
        
        self.fulcon = nn.Sequential(
            # Fully Connected Layer 1
            nn.Linear(
                in_features=12288,
                out_features=1024,
                bias=True
            ),
            nn.Dropout(p=0.4),
            nn.ReLU(),
            
            # Fully Connected Layer 2
            nn.Linear(
                in_features=1024,
                out_features=1024,
                bias=True
            ),
            nn.Dropout(p=0.4),
            nn.ReLU(),
            
            # Regression Layer
            nn.Linear(
                in_features=1024,
                out_features=1,
                bias=True
            )
        )

    def forward(self, batch):
        result = self.convolNet(batch)
        result = result.view(batch.size(0), -1) # reshap pour obtenir une matrice 2 Dims
        result = self.fulcon(result)
        return result
    