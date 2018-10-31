import torch
import  torchvision.models as models
from MobileNetV2 import MobileNetV2

# resnet = models.resnet101().cpu().eval()
resnet = MobileNetV2().cpu().eval()
x = torch.randn((1, 3, 224, 224))

def test():
    y = resnet(x)

if __name__ == '__main__':
    import timeit
    print(timeit.timeit("test", setup="from __main__ import test", number=10000000))
