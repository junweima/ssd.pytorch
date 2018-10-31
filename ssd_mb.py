import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc_mb_v2, coco
import os
from models.MobileNetV2 import MobileNetV2

class SSDMobile(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base MobileNetV2 network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: MobileNetV2 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSDMobile, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        assert (num_classes == 21), "coco config for mbv2 is not suppored yet. (jeremy)"
        self.cfg = (coco, voc_mb_v2)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.mbv2 = base
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(32, 20) # TODO: why should this be 20?
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

        self.adapter = nn.Conv2d(32, 512, 1, 1, 0, bias=False)

    def forward(self, x, apply_mixup=False, mixup_batches=None, mixup_lambda=0.5):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        if apply_mixup and self.phase == 'train':
            assert mixup_batches is not None, 'mixup batches cannot be None for manifold mixup'
            x = self.mbv2.features[:1](x)
            x = self.mbv2.features[1:7](x)
            x_intermediate = x

            # apply manifold mixup
            x1 = x[mixup_batches[0], ...]
            x2 = x[mixup_batches[1], ...]
            x = mixup_lambda * x1 + (1-mixup_lambda) * x2

        elif not apply_mixup and self.phase == 'train':
            # feature map size of [6] =  38^2  * 32
            x = self.mbv2.features[:7](x)
            x_intermediate = None
        else:
            assert self.phase == 'test', 'wrong input'
            # feature map size of [6] =  38^2  * 32
            x = self.mbv2.features[:7](x)
            x_intermediate = None


        # normalization according to the experiment section in ssd paper
        s = self.L2Norm(x)
        sources.append(s)

        # feature map size of 19^2 * 96
        x = self.mbv2.features[7:13](x)
        sources.append(x)

        # feature map size of [-1] = 10^2 * 1280
        x = self.mbv2.features[13:-1](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )

        if self.phase == 'train' and apply_mixup:
            x_intermediate = self.adapter(x_intermediate)
        else:
            x_intermediate = None

        # import pdb; pdb.set_trace()

        return output, x_intermediate

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(mbv2, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    mbv2_source = [6, 13, -2] # see documents/mobilenetV2.md
    for k, v in enumerate(mbv2_source):
        loc_layers += [nn.Conv2d(mbv2.features[v]._modules['conv'][-2].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(mbv2.features[v]._modules['conv'][-2].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], len(mbv2_source)):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return mbv2, extra_layers, (loc_layers, conf_layers)


# extras and mbox: 2 convs less than vgg -> 1 less mbox feature map
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location, determined by config.aspect_ratios
    '512': [],
}


def build_mobile_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    mbv2 = MobileNetV2()
    base_, extras_, head_ = multibox(mbv2,
                                     add_extras(extras[str(size)], 320),
                                     mbox[str(size)], num_classes)
    return SSDMobile(phase, size, base_, extras_, head_, num_classes)


