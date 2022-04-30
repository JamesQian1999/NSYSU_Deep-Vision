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

        self.pool = nn.MaxPool2d(kernel_size = 2, stride=2)


        self.down1 = nn.Sequential(
            #3x240x240
            nn.Conv2d(in_channels = 3, out_channels = 64,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #64x240x240
            nn.Conv2d(in_channels = 64, out_channels = 64,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #64x240x240
            nn.Conv2d(in_channels = 64, out_channels = 64,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #64x240x240
        ) 

        self.down2 = nn.Sequential(
            #64x120x120
            nn.Conv2d(in_channels = 64, out_channels = 128,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #128x120x120
            nn.Conv2d(in_channels = 128, out_channels = 128,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #128x120x120
        )

        self.hidden = nn.Sequential(
            #128x60x60
            nn.Conv2d(in_channels = 128, out_channels = 512,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #512x60x60
            nn.Conv2d(in_channels = 512, out_channels = 512,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #512x60x60
            nn.Conv2d(in_channels = 512, out_channels = 128,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #128x60x60
            nn.ConvTranspose2d(in_channels = 128, out_channels = 128,  kernel_size = 2, stride = 2),
            nn.ReLU(),
            #128x120x120
        )

        self.up1 = nn.Sequential(
            #256x120x120
            nn.Conv2d(in_channels = 256, out_channels = 128,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #128x120x120
            nn.Conv2d(in_channels = 128, out_channels = 128,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #128x120x120
            nn.Conv2d(in_channels = 128, out_channels = 64,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #64x120x120
            nn.ConvTranspose2d(in_channels = 64, out_channels = 64,  kernel_size = 2, stride = 2),
            nn.ReLU(),
            #64x240x240
        ) 

        self.up2 = nn.Sequential(
            #128x240x240
            nn.Conv2d(in_channels = 128, out_channels = 64,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #64x240x240
            nn.Conv2d(in_channels = 64, out_channels = 64,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #64x240x240
            nn.Conv2d(in_channels = 64, out_channels = 64,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #64x240x240
        )

        self.final = nn.Sequential(
            #64x240x240
            nn.Conv2d(in_channels = 64, out_channels = num_classes,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #num_classesx240x240
            #nn.Softmax(dim=1)
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
        x = self.pool(x1)

        x2 = self.down2(x)
        x = self.pool(x2)

        x = self.hidden(x)
        
        xcat =torch.cat((x2,x),dim=1)
        x = self.up1(xcat)

        xcat =torch.cat((x1,x),dim=1)
        x = self.up2(xcat)

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
