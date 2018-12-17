环境：python3+pytorch 0.4.0

预先设置

git clone https://nkwwb@bitbucket.org/detectengine/detectengine.git

cd detectengine/

mkdir data

cd data

ln -s PASCAL_VOC_DEVKIT_PATH VOCdevkit

mkdir pretrained_model

将预训练的model放在pretrained_model下，pretrained model可在 https://github.com/jwyang/faster-rcnn.pytorch/tree/master 下载。

cd ..

cd compiled_modules

sh make.sh  根据显卡计算能力设置make.sh的CUDA_ARCH，保证最高sm与显卡匹配。

cd ..

cd torchlight

python setup.py install

cd ..

训练：

faster rcnn: 

python main.py det_frcnn -c config/detection/frcnn/vgg16\_voc\_1.yaml



yolov2:

python main.py det_yolov2 -c config/detection/yolov2/darknet19\_voc\_1.yaml

不同的配置文件里设置参数，其中文件名中的1，2和里面的session编号一致，表示一种实验设置下的各次实验。



测试：

python main.py det_frcnn -c config/detection/frcnn/vgg16\_voc\_1.yaml --phase test --resume True --dev XX (目前只支持单卡测试)



NOTE：在我机器上，要想设置可见GPU，必须使得os.environ['CUDA_VISIBLE_DEVICES']="0" 先运行，但是这样直接运行的话，不知道为什么占卡速度非常慢，并且会影响到后面的训练/测试速度。要运行“torch.cuda.is_available”这句话速度才正常，但这句一旦放在os.environ前面又会使得不能指定可见GPU。不知道在其他机器上会怎样。我目前的解决方法是另外开一个shell终端，运行python命令，运行torch.cuda.is_available，这就可以了...非常bug.....