"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl

class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=100, hparams=None):
        super().__init__()
        self.hparams.update(hparams)

        self.pool = nn.MaxPool2d(kernel_size = 2, stride=2)


        self.down1 = nn.Sequential(
            #3x240x240
            nn.Conv2d(in_channels = 3, out_channels = 64,  kernel_size = 3, padding = 1),
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

        self.down3 = nn.Sequential(
            #128x60x60
            nn.Conv2d(in_channels = 128, out_channels = 256,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #256x60x60
            nn.Conv2d(in_channels = 256, out_channels = 256,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #256x60x60
        )

        self.hidden = nn.Sequential(
            #256x30x30
            nn.Conv2d(in_channels = 256, out_channels = 1024,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #1024x30x30
            nn.Conv2d(in_channels = 1024, out_channels = 256,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #256x30x30

            nn.ConvTranspose2d(in_channels = 256, out_channels = 256,  kernel_size = 2, stride = 2),
            nn.ReLU(),
            #256x60x60
        )

        self.up1 = nn.Sequential(
            #512x60x60
            nn.Conv2d(in_channels = 512, out_channels = 256,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #256x60x60
            nn.Conv2d(in_channels = 256, out_channels = 128,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #128x60x60

            nn.ConvTranspose2d(in_channels = 128, out_channels = 128,  kernel_size = 2, stride = 2),
            nn.ReLU(),
            #128x120x120
        ) 

        self.up2 = nn.Sequential(
            #256x120x120
            nn.Conv2d(in_channels = 256, out_channels = 128,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #128x120x120
            nn.Conv2d(in_channels = 128, out_channels = 64,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #64x120x120

            nn.ConvTranspose2d(in_channels = 64, out_channels = 64,  kernel_size = 2, stride = 2),
            nn.ReLU(),
            #64x240x240
        )

        self.up3 = nn.Sequential(
            #128x240x240
            nn.Conv2d(in_channels = 128, out_channels = 64,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #64x240x240
            nn.Conv2d(in_channels = 64, out_channels = 64,  kernel_size = 3, padding = 1),
            nn.ReLU(),
            #64x240x240
        )

        self.final = nn.Sequential(
            #64x240x240
            nn.Conv2d(in_channels = 64, out_channels = num_classes,  kernel_size = 3, padding = 1),
            #num_classesx240x240
        )


    def forward(self, x):

        x1 = self.down1(x)
        x = self.pool(x1)

        x2 = self.down2(x)
        x = self.pool(x2)

        x3 = self.down3(x)
        x = self.pool(x3)

        x = self.hidden(x)
        
        xcat =torch.cat((x3,x),dim=1)
        x = self.up1(xcat)

        xcat =torch.cat((x2,x),dim=1)
        x = self.up2(xcat)

        xcat =torch.cat((x1,x),dim=1)
        x = self.up3(xcat)

        x = self.final(x)


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

