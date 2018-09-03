from ssd_mb import *
import torch

x = torch.randn((4, 3, 300, 300))
ssdmb = build_mobile_ssd('train')

y = ssdmb(x)