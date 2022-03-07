import torch.nn as nn
import numpy as np


def simple_conv1d_block(in_planes, out_planes, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=False),
        nn.BatchNorm1d(out_planes),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(kernel_size=4)
    )


def simple_conv2d_block(in_planes, out_planes, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2)
    )


class ResBasicBlock1d(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, outplanes,
                               kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(outplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            outplanes, outplanes, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(outplanes)
        self.relu2 = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride > 1 or inplanes != outplanes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inplanes, outplanes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(outplanes),
            )

    def forward(self, x):
        o = self.conv1(x)
        o = self.bn1(o)
        o = self.relu1(o)
        o = self.conv2(o)
        o = self.bn2(o)
        o = self.relu2(o + self.shortcut(x))
        return o


class ResBasicBlock2d(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes,
                               kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            outplanes, outplanes, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.relu2 = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride > 1 or inplanes != outplanes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(outplanes),
            )

    def forward(self, x):
        o = self.conv1(x)
        o = self.bn1(o)
        o = self.relu1(o)
        o = self.conv2(o)
        o = self.bn2(o)
        o = self.relu2(o + self.shortcut(x))
        return o


def simple_general_cnn(args):
    N = int(np.log(args.audio_length / 10)/np.log(4)) - 1
    M = 2**(N + 5) * int(args.audio_length / (10 * 4 ** (N+1)))
    return nn.Sequential(
        simple_conv1d_block(1, 32, 40, stride=10, padding=15),
        *[simple_conv1d_block(2**(5+i), 2**(6+i), 3, padding=1)
          for i in range(N)],
        nn.Flatten(),
        nn.Linear(M, args.num_classes)
    )


def simple_general_resnet(args):
    N = int(np.log(args.audio_length / 10)/np.log(4)) - 1
    M = 2**(N + 5) * int(args.audio_length / (10 * 4 ** (N+1)))
    return nn.Sequential(
        simple_conv1d_block(1, 32, 40, stride=10, padding=15),
        *[nn.Sequential(ResBasicBlock1d(2**(5+i), 2**(6+i)), nn.MaxPool1d(4))
          for i in range(N)],
        nn.Flatten(),
        nn.Linear(M, args.num_classes)
    )


def create_model(args):
    if args.model == "simple_general_cnn":
        return simple_general_cnn(args)
    elif args.model == "simple_general_cnn":
        return simple_general_resnet(args)
    else:
        return None
