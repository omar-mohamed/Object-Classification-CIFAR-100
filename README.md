# Object-Classification-CIFAR-100
A project utilizing deep learning methods to classify the images in the CIFAR-100 dataset.

# Problem:

Given a blurry image, the task is to classify it into one of the 100 classes in CIFAR-100.

# Dataset:

The CIFAR-100 dataset consists of 60000 32x32 colour images in 100 classes, with 600 images per class. There are 50000 training images and 10000 test images. 

Link: [CIFAR_100_Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

### Dataset Splitting:
- 48k Image as a training set
- 2k Image as a dev set
- 10k as a test set

# Preprocessing:
-Normalized pixel values to be between -1 and 1

# Training Method:

 **Used convolutional neural nets**
![1 vkq0hxdaqv57salxajquxa](https://user-images.githubusercontent.com/6074821/52169534-a6b95780-2742-11e9-9a16-c0fab98bbd1b.jpeg)

 **Used the following architecture:** <br/>
 
 Conv layer 1: 32 filters with size of 7x7. <br/>
 Max Pooling layer: filter size 2x2. <br/>
 BatchNorm layer. <br/>
 Dropout with Keep prob : 0.9. <br/>
 
 Conv layer 2: 64 filters with size of 5x5. <br/>
 Max Pooling layer: filter size 2x2. <br/>
 BatchNorm layer. <br/>
 Dropout with Keep prob : 0.9. <br/>
 
 Conv layer 3: 128 filters with size of 3x3. <br/>
 Max Pooling layer: filter size 2x2. <br/>
 BatchNorm layer. <br/>
 Dropout with Keep prob : 0.9. <br/>
 
 Conv layer 4: 256 filters with size of 1x1. <br/>
 Max Pooling layer: filter size 2x2. <br/>
 BatchNorm layer. <br/>
 Dropout with Keep prob : 0.9. <br/>
 
 Fully connected layer 1: 1024 hidden neurons <br/>
 Fully connected layer 2: 1024 hidden neurons <br/>
 Fully connected layer 3: 1024 hidden neurons <br/>
 
 **Additional details:** <br/>
 Used Adam Optimizer <br/>
 Used learning rate decay <br/>
 Used mini batch of size 32 <br/>
 Used xavier weight initialization <br/>
 Used relu activation in hidden states <br/>
 Used softmax with cross entropy in output <br/>
 Used early stopping <br/>
 
# Results:
 Ongoing..

# Usage:
Here we will discuss how to run the project and what each file is responsible of:

### DownloadData.py:
This script will download the CIFAR-100 dataset.

### ExtractData.py:
This script will extract the CIFAR-100 dataset.

### PreprocessData.py:
This script will load the data, normalize it, shuffle it, take 2k images from training as dev set, and save it in a pickle file.

### TrainModel.py:
This script will begin training on the training data and output the results(including on test set after it finishes training) and save the accuracy and loss graphs in output_images folder, and save the graph info for tensor board in graph_info folder, and save the model itself in saved_model

### TestModel.py:
This script will load the model saved in best_model folder(which gave the best accuracy overall) and run it on the test set and output the results.

### Prediction_Interface.py:
This script will open a gui view for you to load an image and classify it using the model in best_model folder.

# Environment Used:
- Python 3.6.1
- Tensorflow 1.9
