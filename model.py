import torch
import torch.nn as nn
import torch.nn.functional as F

class LPC(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1):
        """
        Initializes YOLOv5 SPPF layer with given channels and kernel size for YOLOv5 model, combining convolution and
        max pooling.

        Equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = self._conv_layer(c1,c_)
        self.cv2 = self._conv_layer(c_,c_)
        self.cv3= self._conv_layer(c1*2,c1*2,1)
    def forward(self, x):
        """Processes input through a series of convolutions and max pooling operations for feature extraction."""
        x1 = self.cv1(x)
        x2=self.cv2(x1)
        x=(torch.cat((x, x1,x2), 1))
        x=self.cv3(x)
        return x
    @staticmethod
    def _conv_layer(in_channels, out_channels,k=3):
        layer = [nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=1, bias=False),
                 nn.BatchNorm2d(out_channels, eps=1e-4),
                 nn.ReLU(inplace=True)]
        return nn.Sequential(*layer)
class LPC_SonoNet(nn.Module):
    def __init__(self,  num_labels=6, in_channels=1):
        super().__init__()
        self.feature_channels = 1024

        self.features=self.feature_layer(in_channels,64)
        self.lpc1=LPC(64)
        self.avg1=self.max_pool()
        self.lpc2=LPC(128)
        self.avg2 = self.max_pool()
        self.lpc3=LPC(256)
        self.avg3 = self.max_pool()
        self.lpc4=LPC(512)

        self.adaption_channels = self.feature_channels // 2
        self.num_labels = num_labels
        self.adaption = self.adaption_layer(self.feature_channels, self.adaption_channels, self.num_labels)


    def forward(self, x):
        x = self.features(x)
        x=self.lpc1(x)
        x=self.avg1(x)
        x=self.lpc2(x)
        x=self.avg2(x)
        x=self.lpc3(x)
        x=self.avg3(x)
        x=self.lpc4(x)
        x = self.adaption(x)
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)
        x = F.softmax(x, dim=1)
        return x

    def feature_layer(cls,in_channels=3,out_channels=64):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def max_pool(cls):
        return nn.MaxPool2d(2)

    def adaption_layer(cls,feature_channels, adaption_channels, num_labels):
        return nn.Sequential(
            nn.Conv2d(feature_channels, adaption_channels, 1, bias=False),
            nn.BatchNorm2d(adaption_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(adaption_channels, num_labels, 1, bias=False),
            nn.BatchNorm2d(num_labels),
        )