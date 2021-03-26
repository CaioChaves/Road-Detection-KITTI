## Road detection on KITTI Images
* [Project overall description](#general-info)
* [Setup](#setup)
* [How to run a piece of code](#how-to-run-a-piece-of-code)

## General info

This project aimed at studying different supervised learning algorithms to perform road detection in images captured from a driving context (KITTI Dataset). It was developped in the context of the author's research internship which took place at ENSTA engineering school between March and July 2019.

![alt text](https://github.com/CaioChaves/Road-Detection-KITTI/blob/main/images-report/illustration.jpg?raw=true)

It can be divided in three main parts:
* in the first one, an ordinary convolutional neural network (CNN) was trained from scratch on the KITTI dataset to classify small patches of the image as road or not-road. Some variations to this base model architecture were proposed, mainly the addition of multi-scale patches and location features, and their effect on accuracy rate was evaluated. 
* In the second part, these CNNs models were used to perform semantic segmentation through a method of sliding window. 
* Finally, the algorithms efficiency were studied, a minimalist and faster version introduced and all the models compared to state-of-the-art network in this field. This study revealed that patch-classification-based algorithms can achieve slightly worse, yet comparable accuracy rate when compared to what the state-of-the-art does. Besides, that algorithm is considerably lighter, which may make it a good choice in some applications for which memory and processing power are limited, such as some embedded systems.

The full report is also available at the repo, providing extra information about this work.

## Setup
To install the required Python packages for this, we recommend creating a virtual environment:

```
$ git clone https://github.com/CaioChaves/Road-Detection-KITTI.git
$ cd Road-Detection-KITTI
$ virtualenv Road-Detection-KITTI
$ source Road-Detection-KITTI/bin/activate
$ pip install -r requirements.txt
```

## How to run a piece of code

To perform road detection on an image, the following command line can be executed:

```
$ python generate_semantic_segmentation.py --arch mini --img-number 0 --stride 10 --epoch-number 100 --model-folder '2019-07-11_17:54:57/'
```

By applying this method to a sequence of images, we are able to detect road on a driving scene like this one:

[![Watch the video](https://img.youtube.com/vi/L4z4fyKgkEs/maxresdefault.jpg)](https://youtu.be/L4z4fyKgkEs)

