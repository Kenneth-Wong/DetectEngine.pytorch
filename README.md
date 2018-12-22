# A *Faster* Pytorch Implementation of Detection Methods (Faster R-CNN+YOLOv2)

## Introduction

This project is a *faster* pytorch implementation which aims at collecting all detection methods, and now it contains faster R-CNN and YOLOv2 temporarily. The faster R-CNN is trained perfectly and the results are satisfying while YOLOv2 is still an experimental project since the results are not good enough. We encourage you to help us debug.  

Recently, there are a number of good implementations which we refer to :

* [rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn), developed based on Pycaffe + Numpy
* [longcw/faster_rcnn_pytorch](https://github.com/longcw/faster_rcnn_pytorch), developed based on Pytorch + Numpy
* [endernewton/tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn), developed based on TensorFlow + Numpy
* [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn), developed based on Pytorch + TensorFlow + Numpy
* [longcw/yolov2-pytorch](https://github.com/longcw/yolo2-pytorch), developed based on Pytorch + Numpy
* [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch), developed based on Pytorch, which is pure  Pytorch code

During our implementing, we referred the above implementations, especailly [longcw/yolov2-pytorch](https://github.com/longcw/yolo2-pytorch) and [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch) which this README mainly referred to. Besides, we reffered the coding style and framework and borrowed some codes from [yysijie/st-gcn](https://github.com/yysijie/st-gcn). Thank all of them very much!  

Our implementation has several unique and new features:

* **It is pure Pytorch code**. We convert all the numpy implementations to pytorch, including faster-RCNN and YOLOv2. 

* **It supports multi-image batch training**. We revise all the layers, including dataloader, rpn, roi-pooling, etc., to support multiple images in each minibatch.

* **It supports multiple GPUs training**. We use a multiple GPU wrapper (nn.DataParallel here) to make it flexible to use one or more GPUs, as a merit of the above two features.

* **It supports three pooling methods**. We integrate three pooling methods: roi pooing, roi align and roi crop. More importantly, we modify all of them to support multi-image batch training.

* **It is memory efficient**. We limit the image aspect ratio, and group images with similar aspect ratios into a minibatch. As such, we can train resnet101 and VGG16 with batchsize = 4 (4 images) on a sigle Titan Xp (12 GB). 

* **It is faster**. Based on the above modifications, the training is much faster. We report the training speed on NVIDIA TITAN Xp in the tables below.

### What we are doing and going to do

- [x] Support both python2 and python3 (but use python3 better).
- [x] Add deformable pooling layer (mainly supported by [Xander](https://github.com/xanderchf)).
- [x] Support pytorch-0.4.0 (this branch).
- [ ] Support tensorboardX.
- [ ] Support Visdom.
- [ ] Support pytorch-1.0 (go to pytorch-1.0 branch).

## Other Implementations

* [Feature Pyramid Network (FPN)](https://github.com/jwyang/fpn.pytorch)

* [Mask R-CNN](https://github.com/roytseng-tw/mask-rcnn.pytorch) (~~ongoing~~ already implemented by [roytseng-tw](https://github.com/roytseng-tw))


## Tutorial

* [Blog](http://www.telesens.co/2018/03/11/object-detection-and-classification-using-r-cnns/) by [ankur6ue](https://github.com/ankur6ue)
* Blog for Faster-RCNN and SIN(CVPR 2018) by [Me](https://www.cnblogs.com/Kenneth-Wong/p/8624647.html)

## Benchmarking

We benchmark our code thoroughly on three datasets: pascal voc, using vgg16 and resnet101 for faster-RCNN and darknet19 for YOLOv2. Below are the results:

1). PASCAL VOC 2007 (Train/Test: 07trainval/07test, frcnn: scale=600, ROI Align.  yolov2: multiscale training, from 320 to 608, reisze every 10 batches)

| model                                    | #GPUs | batch size | lr   | lr_decay | max_epoch | time/epoch | mem/GPU | mAP  |
| ---------------------------------------- | ----- | ---------- | ---- | -------- | --------- | ---------- | ------- | ---- |
| [frcnn-VGG16](https://pan.baidu.com/s/1nghCLduESqdXsEvJBwboZA) | 1     | 1          | 1e-3 | 5        | 6         | 0.52 hr    | 4297MB  | 69.5 |
| [frcnn-VGG16](https://pan.baidu.com/s/1bKYgI6PJXl_GV8Zr2r8XwA) | 1     | 4          | 4e-3 | 8        | 9         | 0.48 hr    | 9415MB  | 69.8 |
| [frcnn-VGG16](https://pan.baidu.com/s/1H_g_Ly85hoThhs2bACzrYQ) | 2     | 8          | 1e-2 | 8,11     | 13        | 0.30 hr    | 8922MB  | 68.9 |
|                                          |       |            |      |          |           |            |         |      |
| [frcnn-Res101](https://pan.baidu.com/s/1pPxijeF-H1JbPRfWhTvOhg) | 1     | 1          | 1e-3 | 5        | 7         | 0.48 hr    | 3503MB  | 75.3 |
| [frcnn-Res101](https://pan.baidu.com/s/1i89tCakV9w4AsDh9K-4L4g) | 1     | 4          | 4e-3 | 8        | 10        | 0.37 hr    | 9869MB  | 74.6 |
| [frcnn-Res101](https://pan.baidu.com/s/1UFyXhEgFPOX5msrOV-_Dog) | 3     | 12         | 1e-2 | 8        | 10        | 0.17 hr    | 9399MB  | 74.8 |
| [frcnn-Res101](https://pan.baidu.com/s/1NT-rND7BulPvLnOKsI7eaw) | 4     | 16         | 1e-2 | 8        | 10        | 0.14 hr    | 9627MB  | 74.3 |
|                                          |       |            |      |          |           |            |         |      |
| [yolov2-dark19](https://pan.baidu.com/s/102U9psRdr9r_CGrtT6MFCg) | 3     | 48         | 1e-3 | 60,90    | 160       | 0.02 hr    | 11000MB | 52.5 |
|                                          |       |            |      |          |           |            |         |      |


2). PASCAL VOC 2007+2012(Train/Test: 0712trainval/07test, frcnn: scale=600, ROI Align.  yolov2: multiscale training, from 320 to 608, reisze every 10 batches)

| model                                    | #GPUs | batch size | lr   | lr_decay | max_epoch | time/epoch | mem/GPU | mAP        |
| ---------------------------------------- | ----- | ---------- | ---- | -------- | --------- | ---------- | ------- | ---------- |
| [yolov2-dark19](https://pan.baidu.com/s/1cOdp8gQrdJVMPdvI8OIkkg) | 3     | 48         | 1e-3 | 60,90    | 160       | 0.07 hr    | 11000MB | 63.2 (416) |

**!! NOTE: The results of YOLOv2-darknet19 is now far from expectation. If you are interested in it, we are glad that you can join us to debug these codes. **

* Click the links in the above tables to download our trained models.
* If not mentioned, the GPU we used is NVIDIA Titan Xp Pascal (12GB).

## Preparation


First of all, clone the code
```
git clone https://github.com/Kenneth-Wong/DetectEngine.pytorch.git DetectEngine
```

Then, create a folder:
```
cd DetectEngine && mkdir data
```

Then, install torchlight module:

```
cd torchlight
python setup.py install
```

**Note: **

### prerequisites

* Python 2.7 or 3.6
* Pytorch 0.4.0 (**now it does not support 0.4.1 or higher**)
* CUDA 8.0 or higher

### Data Preparation

* **PASCAL_VOC 07+12**: Please follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare VOC datasets. Actually, you can refer to any others. After downloading the data, creat softlinks in the folder data/.
* **COCO**: Please also follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare the data. (NOT support temporarily)
* **Visual Genome**: Please follow the instructions in [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) to prepare Visual Genome dataset. You need to download the images and object annotation files first, and then perform proprecessing to obtain the vocabulary and cleansed annotations based on the scripts provided in this repository. (NOT support temporarily)

### Pretrained Model

We used two pretrained models in our experiments, VGG and ResNet101. You can download these two models from:

* VGG16: [Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth)

* ResNet101: [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)

Download them and put them into the data/pretrained_model/.

**NOTE**. We compare the pretrained models from Pytorch and Caffe, and surprisingly find Caffe pretrained models have slightly better performance than Pytorch pretrained. We would suggest to use Caffe pretrained models from the above link to reproduce our results.

**If you want to use pytorch pre-trained models, please remember to transpose images from BGR to RGB, and also use the same data transformer (minus mean and normalize) as used in pretrained model.**

### Compilation

As pointed out by [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn), choose the right `-arch` in `make.sh` file, to compile the cuda code:

| GPU model                  | Architecture |
| -------------------------- | ------------ |
| TitanX (Maxwell/Pascal)    | sm_52        |
| GTX 960M                   | sm_50        |
| GTX 1080 (Ti)              | sm_61        |
| Grid K520 (AWS g2.2xlarge) | sm_30        |
| Tesla K80 (AWS p2.xlarge)  | sm_37        |
| Titan Xp                   | sm_61        |

More details about setting the architecture can be found [here](https://developer.nvidia.com/cuda-gpus) or [here](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

Compile the cuda dependencies using following simple commands:

```
cd compiled_modules
sh make.sh
```

It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Align and ROI_Crop for faster-RCNN and Reorg layer for Yolov2. The default version is compiled with Python 3.6, please compile by yourself if you are using a different python version.

**As pointed out in this [issue](https://github.com/jwyang/faster-rcnn.pytorch/issues/16), if you encounter some error during the compilation, you might miss to export the CUDA paths to your environment.**

## Train

Both the training and test parameters can be set in config file: for example, DetectEngine/config/frcnn(or yolov2)/vgg16\_voc\_1.yaml. In this file, you can set your working root direction, detection framework, session id (the experiment number), training dataset, GPU ids, etc. The priority of paramters is **Command lines > config files > default**. Before training and test, set parameters according to your environment. We also provide a config template.

For example, to train a faster R-CNN model with vgg16 on pascal_voc 2007, simply run:
```
python main.py det_frcnn -c ./config/frcnn/vgg16_voc_1.yaml
```
## Test

**Now we only support single GPU test**. If you want to evaluate the detection performance of the final trained  model on pascal_voc test set, simply run
```
python main.py det_frcnn -c ./config/frcnn/vgg16_voc_1.yaml --resume True --device 0 --phase test
```
It will automatically load the latest model from current working root. Besides, you can also determine any model you want to evaluate in the config file by setting the checkpoint direction for "checkpoint" parameter or just run the test command like this:

```
python main.py det_frcnn -c ./config/frcnn/vgg16_voc_1.yaml --resume True --device 0 --phase test --checkpoint CHECKPOINT_FILE
```



## Citation

    @article{jjfaster2rcnn,
        Author = {Jianwei Yang and Jiasen Lu and Dhruv Batra and Devi Parikh},
        Title = {A Faster Pytorch Implementation of Faster R-CNN},
        Journal = {https://github.com/jwyang/faster-rcnn.pytorch},
        Year = {2017}
    }
    
    @inproceedings{renNIPS15fasterrcnn,
        Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
        Title = {Faster {R-CNN}: Towards Real-Time Object Detection
                 with Region Proposal Networks},
        Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
        Year = {2015}
    }
    
    @inproceedings{redmon2017yolo9000,
      title={YOLO9000: Better, Faster, Stronger},
      author={Redmon, Joseph and Farhadi, Ali},
      booktitle={2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      pages={6517--6525},
      year={2017},
      organization={IEEE}
    }
    
    @inproceedings{stgcn2018aaai,
      title     = {Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition},
      author    = {Sijie Yan and Yuanjun Xiong and Dahua Lin},
      booktitle = {AAAI},
      year      = {2018},
    }
    
