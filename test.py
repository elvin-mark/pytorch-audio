from models import simple_general_resnet, simple_general_cnn

import torch


class Test:
    def __init__(self):
        self.num_classes = 10
        self.audio_length = 18000


model = simple_general_resnet(Test())
print(model)
print(model(torch.zeros(1, 1, 18000)).shape)
