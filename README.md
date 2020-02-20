# Detecting Building license plate using TensorFlow Object Detection
## Brief Summary
***last updated: 20/02/2020 with Tensorflow v1.15***  
  
This repertory is a deep learning project that creates models that recognize building license plates as part of Inha University's Winter School project. This project ran on Windows, and all configuration is Windows based. Participants in this project are Lee Joo-ho and Park Ki-soo.  
  
This repository only covers creating models that recognize building plates using Python code on PC, and the rest of the process is covered elsewhere. This repository provides all the files needed to train building plates detector.  
  
This readme describes every steps required to get going with obect detection classifier.  
It also describes how to replace the files with your own files to train a detection classifier for whatever you want. It also has Python scripts to test your classifier out on an image, video, or webcam feed.  
  
This project is inspired by "[How To Train an Object Detection Classifier for Multiple Objects Using TensorFlow (GPU) on Windows 10](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#1-install-anaconda-cuda-and-cudnn)"  
  

## Introduction

The purpose of this project is to create a tensorflow model that can detects building license plate from the image, to make an application that automatically recognizes addresses when photos are taken.  
  
The reason why we need to train a model that recognizes a license plate is that in general, when a picture is taken, it most likely contains other letters besides the license plate.  
So we should recognize the license plate first, and then recognize the letters.  
In this progress, we will create a model that recognizes building license plates using TensorFlow Object Detection api.  
  
We will use TensorFlow-GPU v1.15. TensorFlow-GPU allows your PC to use the video card to provide extra processing power while training.  
You could use only cpu to train the model but it would take much longer times. If you use CPU-only TensorFlow, you don't need to install CUDA and cuDNN in Step 1.  
  
## Steps
1. Install Anaconda, CUDA, and cuDNN  
