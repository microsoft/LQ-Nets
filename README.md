# LQ-Nets

By [Dongqing Zhang](https://github.com/zdqzeros), [Jiaolong Yang](http://jlyang.org), [Dongqiangzi Ye](https://github.com/EowinYe), [Gang Hua](https://www.microsoft.com/en-us/research/people/ganghua/).

Microsoft Research Asia (MSRA).

### Introduction
This repository contains the training code of LQ-Nets introduced in our ECCV 2018 paper:

D. Zhang*, J. Yang*, D. Ye* and G. Hua. LQ-Nets: Learned Quantization for Highly Accurate and Compact Deep Neural Networks. ECCV 2018 (*: Equal contribution) [PDF](https://arxiv.org/pdf/1807.10029.pdf)

### Dependencies

+ Python 2.7 or 3.3+
+ Python bindings for OpenCV
+ TensorFlow >= 1.3.0
+ [TensorPack](https://github.com/tensorpack/tensorpack)

### Usage

Download the ImageNet dataset and decompress into the structure like

    dir/
      train/
        n01440764/
          n01440764_10026.JPEG
          ...
        ...
      val/
        ILSVRC2012_val_00000001.JPEG
        ...

To train a quantized "pre-activation" ResNet-18, simply run

    python imagenet.py --gpu 0,1,2,3 --data /PATH/TO/IMAGENET --mode preact --depth 18 --qw 1 --qa 2 --logdir_id w1a2 

After the training, the result model will be stored in `./train_log/w1a2`.

For more options, please refer to `python imagenet.py -h`. 

### Results
**ImageNet Experiments**

Quantizing both weight and activation

Model|Bit-width(W/A)|Top-1(%)|Top-5(%)
:---:|:---:|:---:|:---:
ResNet-18|1/2|62.6|84.3
ResNet-18|2/2|64.9|85.9
ResNet-18|3/3|68.2|87.9
ResNet-18|4/4|69.3|88.8
ResNet-34|1/2|66.6|86.9
ResNet-34|2/2|69.8|89.1
ResNet-34|3/3|71.9|90.2
ResNet-50|1/2|68.7|88.4
ResNet-50|2/2|71.5|90.3
ResNet-50|3/3|74.2|91.6
ResNet-50|4/4|75.1|92.4
AlexNet|1/2|55.7|78.8
AlexNet|2/2|57.4|80.1
DenseNet-121|2/2|69.6|89.1
VGG-Variant|1/2|67.1|87.6
VGG-Variant|2/2|68.8|88.6
GoogLeNet-Variant|1/2|65.6|86.4
GoogLeNet-Variant|2/2|68.2|88.1

Quantizing weight only

Model|Bit-width(W/A)|Top-1(%)|Top-5(%)
:---:|:---:|:---:|:---:
ResNet-18|2/32|68.0|88.0
ResNet-18|3/32|69.3|88.8
ResNet-18|4/32|70.0|89.1
ResNet-50|2/32|75.1|92.3
ResNet-50|4/32|76.4|93.1
AlexNet|2/32|60.5|82.7

More results can be found in the paper.

### Citation
If you use our code or models in your research, please cite our paper with

    @inproceedings{ZhangYangYeECCV2018,
        author = {Zhang, Dongqing and Yang, Jiaolong and Ye, Dongqiangzi and Hua, Gang},
        title = {LQ-Nets: Learned Quantization for Highly Accurate and Compact Deep Neural Networks},
        booktitle = {European Conference on Computer Vision (ECCV)},
        year = {2018}
    }
