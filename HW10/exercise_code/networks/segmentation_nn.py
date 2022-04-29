"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        #self.hparams = hparams
        self.hparams.update(hparams)
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        self.pool = nn.MaxPool2d(kernel_size = 2, stride=2, return_indices=True)

        self.unpool = nn.MaxUnpool2d(kernel_size = 2, stride=2)

        self.down1 = nn.Sequential(
            #3x240x240
            nn.Conv2d(in_channels = 3, out_channels = 32,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #32x240x240
            nn.Conv2d(in_channels = 32, out_channels = 32,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #32x240x240
            nn.Conv2d(in_channels = 32, out_channels = 32,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #32x240x240
        ) 

        self.down2 = nn.Sequential(
            #32x120x120
            nn.Conv2d(in_channels = 32, out_channels = 64,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #64x120x120
            nn.Conv2d(in_channels = 64, out_channels = 64,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #64x120x120
        )

        self.down3 = nn.Sequential(
            #64x60x60
            nn.Conv2d(in_channels = 64, out_channels = 128,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #128x60x60
        )

        self.hidden = nn.Sequential(
            #128x30x30
            nn.Conv2d(in_channels = 128, out_channels = 512,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #512x30x30
            nn.Conv2d(in_channels = 512, out_channels = 512,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #512x30x30
            nn.Conv2d(in_channels = 512, out_channels = 128,  kernel_size = 3, padding = 1),
            nn.ReLU()
            #128x30x30
        )

        self.up1 = nn.Sequential(
            # unpool 128x30x30 => 128x60x60
            # cat    128x60x60 => 256x60x60
            nn.Conv2d(in_channels = 256, out_channels = 128,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #128x60x60
            nn.Conv2d(in_channels = 128, out_channels = 128,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #128x60x60
            nn.Conv2d(in_channels = 128, out_channels = 64,  kernel_size = 3, padding = 1),
            nn.ReLU()
            #64x60x60
        )

        self.up2 = nn.Sequential(
            # unpool 64x60x60 => 64x120x120
            # cat    64x120x120 => 128x120x120
            nn.Conv2d(in_channels = 128, out_channels = 64,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #64x120x120
            nn.Conv2d(in_channels = 64, out_channels = 64,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #64x120x120
            nn.Conv2d(in_channels = 64, out_channels = 32,  kernel_size = 3, padding = 1),
            nn.ReLU()
            #32x120x120
        ) 

        self.up3 = nn.Sequential(
            # unpool 32x120x120 => 32x240x240
            # cat    32x240x240 => 64x240x240
            nn.Conv2d(in_channels = 64, out_channels = 32,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #32x240x240
            nn.Conv2d(in_channels = 32, out_channels = 32,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #32x240x240
            nn.Conv2d(in_channels = 32, out_channels = 32,  kernel_size = 3, padding = 1),
            nn.ReLU()
            #32x240x240
        )

        self.final = nn.Sequential(
            #32x240x240
            nn.Conv2d(in_channels = 32, out_channels = num_classes,  kernel_size = 3, padding = 1),
            nn.ReLU()
            #num_classesx240x240
        )


        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        x1 = self.down1(x)
        x, indices1 = self.pool(x1)

        x2 = self.down2(x)
        x, indices2 = self.pool(x2)

        x3 = self.down3(x)
        x, indices3 = self.pool(x3)

        x = self.hidden(x)
        
        x = self.unpool(x,indices3)
        xcat =torch.cat((x,x3),dim=1)
        x = self.up1(xcat)

        x = self.unpool(x,indices2)
        xcat =torch.cat((x,x2),dim=1)
        x = self.up2(xcat)

        x = self.unpool(x,indices1)
        xcat = torch.cat((x,x1),dim=1)
        x = self.up3(xcat)

        x = self.final(x)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
