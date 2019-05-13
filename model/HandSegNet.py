import torch.nn as nn
import math
import torch.utils.model_zoo
from base import BaseModel

def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=3, stride=stride, padding=1, bias=True)


class HandSegNet(BaseModel):
    
    def __init__(self, in_channel=1, out_channel=1, stride=1, downsample=None):
        super(HandSegNet, self).__init__()
        self.conv1 = conv3x3(3, 64)
        self.conv2 = conv3x3(64, 64)
        self.maxpool =   nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        
        self.conv3 = conv3x3(64, 128)
        self.conv4 = conv3x3(128, 128)
        self.conv5 = conv3x3(128, 256)
        self.conv6 = conv3x3(256, 256)
        self.conv7 = conv3x3(256, 256)
        self.conv8 = conv3x3(256, 256)
        self.conv9 = conv3x3(256, 512)
        self.conv10 = conv3x3(512, 512)
        self.conv11 = conv3x3(512, 512)
        self.conv12 = conv3x3(512, 512)
        self.conv13 = conv3x3(512, 512)
        self.conv1x1 = nn.Conv2d(512, 2, kernel_size=1, bias=True)
        
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=8)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.maxpool(x)
        
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.maxpool(x)
        
        x = self.relu(self.conv9(x))
        x = self.relu(self.conv10(x))
        x = self.relu(self.conv11(x))
        x = self.relu(self.conv12(x))
        x = self.relu(self.conv13(x))
        
        x = self.conv1x1(x)
        self.out = self.upsample(x)
        # self.mask_hand = torch.argmax(self.out, dim=1)
        return self.out


if __name__ == "__main__":
    model = HandSegNet()
    x = torch.randn(1, 3, 224, 224)
    print(model)
    print(model(x))