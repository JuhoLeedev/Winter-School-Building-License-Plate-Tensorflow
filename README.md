# Detecting Building license plate using TensorFlow Object Detection
## Brief Summary
***last updated: 20/02/2020 with Tensorflow v1.15***  
  
This repertory is a deep learning project that creates models that recognize building license plates as part of Inha University's Winter School project. This project ran on Windows, and all configuration is Windows based. Participants in this project are Lee Ju-ho and Park Ki-soo.  
  
This repository only covers creating models that recognize building plates using Python code on PC, and the rest of the process is covered elsewhere. This repository provides all the files needed to train building plates detector.  
  
This readme describes every steps required to get going with obect detection classifier.  
It also describes how to replace the files with your own files to train a detection classifier for whatever you want. It also has Python scripts to test your classifier out on an image, video, or webcam feed.  
  
This project is inspired by "[How To Train an Object Detection Classifier for Multiple Objects Using TensorFlow (GPU) on Windows 10](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#1-install-anaconda-cuda-and-cudnn)"  
  
![image1](./docs/image1.png)  
  
## Introduction

The purpose of this project is to create a tensorflow model that can detects building license plate from the image, to make an application that automatically recognizes addresses when photos are taken.  
  
The reason why we need to train a model that recognizes a license plate is that in general, when a picture is taken, it most likely contains other letters besides the license plate. So we should recognize the license plate first, and then recognize the letters.  
In this progress, we will create a model that recognizes building license plates using TensorFlow Object Detection api.  
  
We will use TensorFlow-GPU v1.15. TensorFlow-GPU allows your PC to use the video card to provide extra processing power while training.  
You could use only cpu to train the model but it would take much longer times. If you use CPU-only TensorFlow, you don't need to install CUDA and cuDNN in Step 1.  
  
## Steps
### 1. Install Anaconda, CUDA, and cuDNN
First, you should install Anaconda. You can get [here](https://www.anaconda.com/distribution/). Click Python 3.7 version download and run the exe.  
  
Set the file path to install, and click next.  
  
![image2](./docs/image2.png)  
  
Check all the checkbox and click install.  
  
![image3](./docs/image3.png)  
<br>

After installing Anaconda, you should install CUDA and cuDNN. To use CUDA, you must have a NVIDIA graphics card that supports CUDA. You can check that your graphics card supports CUDA or not at [here](https://www.geforce.com/hardware/technology/cuda/supported-gpus?field_gpu_type_value=all). If you don't have CUDA supported graphics card, you have to use TensorFlow-CPU only. Then you can skip this step.  
  
If you checked your graphics card supports CUDA, now you should install CUDA 10.1 version. Make sure that you installed 10.1 version instead 10.2 version(latest version). At the time of this writing TensorFlow do not support 10.2 CUDA. Check the version of CUDA that TensorFlow supports. You can get CUDA 10.1 [here](https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal).  
I recommend you to download local version. Network version may fail during installation. If you have failed during install, it 
might have conflicted with graphics driver. If you failed, remove your graphics driver and try again.  
  
After installing CUDA, get cuDNN [here](https://developer.nvidia.com/rdp/cudnn-archive). To get cuDNN, you have to join NVIDIA membership. You can download cuDNN after sign in the site. Download cuDNN and extract it to your CUDA installed path. If you installed CUDA at default path, it would be here "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1".
  
### 2. Install TensorFlow and Anaconda environment
