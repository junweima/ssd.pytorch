## TODO list
  - [ ] check todo list: L2 norm
  - [x] train mb on pascal
  - [x] train vanilla distillation
  - [x] train knowledgify
  - [ ] feature map matching for pretraining
  - [ ] check the number of bounding boxes predicted


## mbv2 vanilla kd for voc2007, #steps=5k
<pre>
AP for aeroplane = 0.4698
AP for bicycle = 0.4652
AP for bird = 0.4024
AP for boat = 0.2098
AP for bottle = 0.1276
AP for bus = 0.5447
AP for car = 0.5571
AP for cat = 0.6314
AP for chair = 0.1936
AP for cow = 0.2941
AP for diningtable = 0.3454
AP for dog = 0.5634
AP for horse = 0.6083
AP for motorbike = 0.4973
AP for person = 0.4818
AP for pottedplant = 0.1356
AP for sheep = 0.3062
AP for sofa = 0.3900
AP for train = 0.5787
AP for tvmonitor = 0.3909
Mean AP = 0.4097
~~~~~~~~
Results:
0.470
0.465
0.402
0.210
0.128
0.545
0.557
0.631
0.194
0.294
0.345
0.563
0.608
0.497
0.482
0.136
0.306
0.390
0.579
0.391
0.410
~~~~~~~~
</pre>


## mbv2 base network for voc2007, #steps=115k
<pre>
AP for aeroplane = 0.6978
AP for bicycle = 0.7376
AP for bird = 0.5395
AP for boat = 0.4980
AP for bottle = 0.2099
AP for bus = 0.7193
AP for car = 0.7194
AP for cat = 0.7962
AP for chair = 0.3492
AP for cow = 0.6243
AP for diningtable = 0.6840
AP for dog = 0.7590
AP for horse = 0.8096
AP for motorbike = 0.7370
AP for person = 0.6293
AP for pottedplant = 0.3196
AP for sheep = 0.6142
AP for sofa = 0.7165
AP for train = 0.7761
AP for tvmonitor = 0.5842
Mean AP = 0.6260
~~~~~~~~
Results:
0.698
0.738
0.539
0.498
0.210
0.719
0.719
0.796
0.349
0.624
0.684
0.759
0.810
0.737
0.629
0.320
0.614
0.716
0.776
0.584
0.626
~~~~~~~~
</pre>


## mbv2 base network for voc 2007, #steps=75k
<pre>
AP for aeroplane = 0.6515
AP for bicycle = 0.7008
AP for bird = 0.4873
AP for boat = 0.4413
AP for bottle = 0.1932
AP for bus = 0.6833
AP for car = 0.6896
AP for cat = 0.7884
AP for chair = 0.2930
AP for cow = 0.5879
AP for diningtable = 0.6600
AP for dog = 0.7137
AP for horse = 0.7783
AP for motorbike = 0.7021
AP for person = 0.6021
AP for pottedplant = 0.2934
AP for sheep = 0.5610
AP for sofa = 0.6901
AP for train = 0.7440
AP for tvmonitor = 0.5362
Mean AP = 0.5899
~~~~~~~~
Results:
0.652
0.701
0.487
0.441
0.193
0.683
0.690
0.788
0.293
0.588
0.660
0.714
0.778
0.702
0.602
0.293
0.561
0.690
0.744
0.536
0.590
~~~~~~~~
</pre>

## mbv2 base network for voc2007, #steps=25k
<pre>
AP for aeroplane = 0.5934
AP for bicycle = 0.6147
AP for bird = 0.4668
AP for boat = 0.4075
AP for bottle = 0.1614
AP for bus = 0.6427
AP for car = 0.6325
AP for cat = 0.7778
AP for chair = 0.2636
AP for cow = 0.5466
AP for diningtable = 0.5963
AP for dog = 0.6785
AP for horse = 0.7603
AP for motorbike = 0.6197
AP for person = 0.5619
AP for pottedplant = 0.2259
AP for sheep = 0.5219
AP for sofa = 0.6461
AP for train = 0.7401
AP for tvmonitor = 0.4979
Mean AP = 0.5478
~~~~~~~~
Results:
0.593
0.615
0.467
0.407
0.161
0.643
0.633
0.778
0.264
0.547
0.596
0.678
0.760
0.620
0.562
0.226
0.522
0.646
0.740
0.498
0.548
~~~~~~~~
</pre>

## vgg base network for voc2007
<pre>
AP for aeroplane = 0.8172
AP for bicycle = 0.8544
AP for bird = 0.7572
AP for boat = 0.6958
AP for bottle = 0.4990
AP for bus = 0.8480
AP for car = 0.8572
AP for cat = 0.8737
AP for chair = 0.6142
AP for cow = 0.8233
AP for diningtable = 0.7917
AP for dog = 0.8559
AP for horse = 0.8709
AP for motorbike = 0.8474
AP for person = 0.7893
AP for pottedplant = 0.5000
AP for sheep = 0.7745
AP for sofa = 0.7912
AP for train = 0.8616
AP for tvmonitor = 0.7631
Mean AP = 0.7743
~~~~~~~~
Results:
0.817
0.854
0.757
0.696
0.499
0.848
0.857
0.874
0.614
0.823
0.792
0.856
0.871
0.847
0.789
0.500
0.774
0.791
0.862
0.763
0.774
~~~~~~~~

</pre>

## vgg base network on voc 2007, trained from initialization, #steps=30k
<pre>
AP for aeroplane = 0.7655
AP for bicycle = 0.8044
AP for bird = 0.7220
AP for boat = 0.6612
AP for bottle = 0.4301
AP for bus = 0.7934
AP for car = 0.8286
AP for cat = 0.8655
AP for chair = 0.5367
AP for cow = 0.7415
AP for diningtable = 0.6844
AP for dog = 0.8104
AP for horse = 0.8251
AP for motorbike = 0.7923
AP for person = 0.7572
AP for pottedplant = 0.4366
AP for sheep = 0.7086
AP for sofa = 0.7496
AP for train = 0.8410
AP for tvmonitor = 0.7240
Mean AP = 0.7239
~~~~~~~~
Results:
0.765
0.804
0.722
0.661
0.430
0.793
0.829
0.865
0.537
0.742
0.684
0.810
0.825
0.792
0.757
0.437
0.709
0.750
0.841
0.724
0.724
~~~~~~~~

</pre>


## vgg base network on voc 2007, trained from initialization, #steps=90k
<pre>
P for aeroplane = 0.7966
AP for bicycle = 0.8298
AP for bird = 0.7468
AP for boat = 0.7191
AP for bottle = 0.4937
AP for bus = 0.8537
AP for car = 0.8599
AP for cat = 0.8851
AP for chair = 0.6089
AP for cow = 0.8301
AP for diningtable = 0.7662
AP for dog = 0.8497
AP for horse = 0.8696
AP for motorbike = 0.8312
AP for person = 0.7844
AP for pottedplant = 0.4996
AP for sheep = 0.7884
AP for sofa = 0.7885
AP for train = 0.8494
AP for tvmonitor = 0.7618
Mean AP = 0.7706
~~~~~~~~
Results:
0.797
0.830
0.747
0.719
0.494
0.854
0.860
0.885
0.609
0.830
0.766
0.850
0.870
0.831
0.784
0.500
0.788
0.789
0.849
0.762
0.771
~~~~~~~~
</pre>

# SSD: Single Shot MultiBox Object Detector, in PyTorch
A [PyTorch](http://pytorch.org/) implementation of [Single Shot MultiBox Detector](http://arxiv.org/abs/1512.02325) from the 2016 paper by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang, and Alexander C. Berg.  The official and original Caffe code can be found [here](https://github.com/weiliu89/caffe/tree/ssd).


<img align="right" src= "https://github.com/amdegroot/ssd.pytorch/blob/master/doc/ssd.png" height = 400/>

### Table of Contents
- <a href='#installation'>Installation</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training-ssd'>Train</a>
- <a href='#evaluation'>Evaluate</a>
- <a href='#performance'>Performance</a>
- <a href='#demos'>Demos</a>
- <a href='#todo'>Future Work</a>
- <a href='#references'>Reference</a>

&nbsp;
&nbsp;
&nbsp;
&nbsp;

## Installation
- Install [PyTorch](http://pytorch.org/) by selecting your environment on the website and running the appropriate command.
- Clone this repository.
  * Note: We currently only support Python 3+.
- Then download the dataset by following the [instructions](#datasets) below.
- We now support [Visdom](https://github.com/facebookresearch/visdom) for real-time loss visualization during training!
  * To use Visdom in the browser:
  ```Shell
  # First install Python server and client
  pip install visdom
  # Start the server (probably in a screen or tmux)
  python -m visdom.server
  ```
  * Then (during training) navigate to http://localhost:8097/ (see the Train section below for training details).
- Note: For training, we currently support [VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and [COCO](http://mscoco.org/), and aim to add [ImageNet](http://www.image-net.org/) support soon.

## Datasets
To make things easy, we provide bash scripts to handle the dataset downloads and setup for you.  We also provide simple dataset loaders that inherit `torch.utils.data.Dataset`, making them fully compatible with the `torchvision.datasets` [API](http://pytorch.org/docs/torchvision/datasets.html).


### COCO
Microsoft COCO: Common Objects in Context

##### Download COCO 2014
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/COCO2014.sh
```

### VOC Dataset
PASCAL VOC: Visual Object Classes

##### Download VOC2007 trainval & test
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

##### Download VOC2012 trainval
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```

## Training SSD
- First download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at:              https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
- By default, we assume you have downloaded the file in the `ssd.pytorch/weights` dir:

```Shell
mkdir weights
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

- To train SSD using the train script simply specify the parameters listed in `train.py` as a flag or manually change them.

```Shell
python train.py
```

- Note:
  * For training, an NVIDIA GPU is strongly recommended for speed.
  * For instructions on Visdom usage/installation, see the <a href='#installation'>Installation</a> section.
  * You can pick-up training from a checkpoint by specifying the path as one of the training parameters (again, see `train.py` for options)

## Evaluation
To evaluate a trained network:

```Shell
python eval.py
```

You can specify the parameters listed in the `eval.py` file by flagging them or manually changing them.  


<img align="left" src= "https://github.com/amdegroot/ssd.pytorch/blob/master/doc/detection_examples.png">

## Performance

#### VOC2007 Test

##### mAP

| Original | Converted weiliu89 weights | From scratch w/o data aug | From scratch w/ data aug |
|:-:|:-:|:-:|:-:|
| 77.2 % | 77.26 % | 58.12% | 77.43 % |

##### FPS
**GTX 1060:** ~45.45 FPS

## Demos

### Use a pre-trained SSD network for detection

#### Download a pre-trained network
- We are trying to provide PyTorch `state_dicts` (dict of weight tensors) of the latest SSD model definitions trained on different datasets.  
- Currently, we provide the following PyTorch models:
    * SSD300 trained on VOC0712 (newest PyTorch weights)
      - https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth
    * SSD300 trained on VOC0712 (original Caffe weights)
      - https://s3.amazonaws.com/amdegroot-models/ssd_300_VOC0712.pth
- Our goal is to reproduce this table from the [original paper](http://arxiv.org/abs/1512.02325)
<p align="left">
<img src="http://www.cs.unc.edu/~wliu/papers/ssd_results.png" alt="SSD results on multiple datasets" width="800px"></p>

### Try the demo notebook
- Make sure you have [jupyter notebook](http://jupyter.readthedocs.io/en/latest/install.html) installed.
- Two alternatives for installing jupyter notebook:
    1. If you installed PyTorch with [conda](https://www.continuum.io/downloads) (recommended), then you should already have it.  (Just  navigate to the ssd.pytorch cloned repo and run):
    `jupyter notebook`

    1. If using [pip](https://pypi.python.org/pypi/pip):

```Shell
# make sure pip is upgraded
pip3 install --upgrade pip
# install jupyter notebook
pip install jupyter
# Run this inside ssd.pytorch
jupyter notebook
```

- Now navigate to `demo/demo.ipynb` at http://localhost:8888 (by default) and have at it!

### Try the webcam demo
- Works on CPU (may have to tweak `cv2.waitkey` for optimal fps) or on an NVIDIA GPU
- This demo currently requires opencv2+ w/ python bindings and an onboard webcam
  * You can change the default webcam in `demo/live.py`
- Install the [imutils](https://github.com/jrosebr1/imutils) package to leverage multi-threading on CPU:
  * `pip install imutils`
- Running `python -m demo.live` opens the webcam and begins detecting!

## TODO
We have accumulated the following to-do list, which we hope to complete in the near future
- Still to come:
  * [x] Support for the MS COCO dataset
  * [ ] Support for SSD512 training and testing
  * [ ] Support for training on custom datasets

## Authors

* [**Max deGroot**](https://github.com/amdegroot)
* [**Ellis Brown**](http://github.com/ellisbrown)

***Note:*** Unfortunately, this is just a hobby of ours and not a full-time job, so we'll do our best to keep things up to date, but no guarantees.  That being said, thanks to everyone for your continued help and feedback as it is really appreciated. We will try to address everything as soon as possible.

## References
- Wei Liu, et al. "SSD: Single Shot MultiBox Detector." [ECCV2016]((http://arxiv.org/abs/1512.02325)).
- [Original Implementation (CAFFE)](https://github.com/weiliu89/caffe/tree/ssd)
- A huge thank you to [Alex Koltun](https://github.com/alexkoltun) and his team at [Webyclip](webyclip.com) for their help in finishing the data augmentation portion.
- A list of other great SSD ports that were sources of inspiration (especially the Chainer repo):
  * [Chainer](https://github.com/Hakuyume/chainer-ssd), [Keras](https://github.com/rykov8/ssd_keras), [MXNet](https://github.com/zhreshold/mxnet-ssd), [Tensorflow](https://github.com/balancap/SSD-Tensorflow)
