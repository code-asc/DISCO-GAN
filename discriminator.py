import torch.nn as nn
import torch.nn.functional as F
import torch

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_layer_1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)
        self.relu_layer_1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv_layer_2 = nn.Conv2d(64, 64*2, 4, 2, 1, bias=False)
        self.bn_layer_2 = nn.BatchNorm2d(64*2)
        self.relu_layer_2= nn.LeakyReLU(0.2, inplace=True)

        self.conv_layer_3 = nn.Conv2d(64*2, 64*2*2, 4, 2, 1, bias=False)
        self.bn_layer_3 = nn.BatchNorm2d(64*2*2)
        self.relu_layer_3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv_layer_4 = nn.Conv2d(64*2*2, 64*2*2*2, 4, 2, 1, bias=False)
        self.bn_layer_4 = nn.BatchNorm2d(64*2*2*2)
        self.relu_layer_4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv_layer_5 = nn.Conv2d(64*2*2*2, 1, 4, 1, 0, bias=False)

    def forward(self, x):
        conv_1 = self.conv_layer_1(x)
        relu_layer_1 = self.relu_layer_1(conv_1)

        conv_2 = self.conv_layer_2(relu_layer_1)
        batch_2 = self.bn_layer_2(conv_2)
        relu_layer_2 = self.relu_layer_2(batch_2)

        conv_3 = self.conv_layer_3(relu_layer_2)
        batch_3 = self.bn_layer_3(conv_3)
        relu_layer_3 = self.relu_layer_3(batch_3)

        conv_4 = self.conv_layer_4(relu_layer_3)
        batch_4 = self.bn_layer_4(conv_4)
        relu_layer_4 = self.relu_layer_4(batch_4)

        return torch.sigmoid(self.conv_layer_5(relu_layer_4))
