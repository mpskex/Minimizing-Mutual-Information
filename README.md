# Learning Compact Binary Representation with Unsupervised Deep Neural Network

    Master Thesis at the University of British Columbia Okanagan
    Master of Applied Science 
    Fangrui Liu 2020

[](https://raw.githubusercontent.com/mpskex/Convolutional-Pose-Machine-tf/master/demo/arch.png)

Including implementation for 2 papers:

- Normalized Hash: Learning Unsupervised Binary Representation with Large Batch

- Minimizing Mutual Information for Unsupervised Binary Representation

Note:

- Training scripts are using fixed seed for randomized behaviors.

- Precomputed feature are used to train the hash network. So you need to collect features from datasets before you train the hash networks.

- you can download the ImageNet dataset [here](https://github.com/thuml/HashNet/tree/master/caffe) and please extract the file in ./data/imagenet
