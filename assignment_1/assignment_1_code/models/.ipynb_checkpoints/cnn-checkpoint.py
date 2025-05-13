from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch


class TinyCNN(nn.Module):
    def __init__(self, num_classes=10, image_size=(32,32)):
        super().__init__()
        self.input_size = image_size

        # First convolutional block
        self.conv_block1=nn.Sequential( 
            nn.Conv2d(3, 16, kernel_size=4, padding=2), #padding for getting the edge details
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Second convolutional block
        self.conv_block2=nn.Sequential( 
            nn.Conv2d(16, 32, kernel_size=4, padding=2), #padding for getting the edge details
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Third convolutional block
        self.conv_block3=nn.Sequential( 
            nn.Conv2d(32, 64, kernel_size=4, padding=2), #padding for getting the edge details
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
            
        flat_size= self.calc_flat_size(self.input_size)

        # Fully connected layers
        self.fc1 = nn.Linear(flat_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def calc_flat_size(self, input_size):
        """
        Calculate size for linear layer
        """

        x = torch.zeros(1, 3, *input_size)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = x.view(x.size(0), -1)
        return x.size(1)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)

        
        return x
    