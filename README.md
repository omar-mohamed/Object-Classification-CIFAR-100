# Object-Classification-CIFAR-100
A project utilizing deep learning methods to classify the images in the CIFAR-100 dataset.

# Problem:

Given a blurry image, the task is to classify it into one of the 100 classes in CIFAR-100.

# Dataset:

The CIFAR-100 dataset consists of 60000 32x32 colour images in 100 classes, with 600 images per class. There are 50000 training images and 10000 test images. 

Link: [CIFAR_100_Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

![plt](https://user-images.githubusercontent.com/6074821/52181190-11789a80-27f8-11e9-8104-7751bfce2e18.png)

### Dataset Splitting:
- 50k Image as a training set
- 2k Image as a dev set (a subset of the test set)
- 10k as a test set

# Preprocessing:
- Normalized pixel values to be between -1 and 1

# Training Method:

 **Used convolutional neural nets**
![1 vkq0hxdaqv57salxajquxa](https://user-images.githubusercontent.com/6074821/52169534-a6b95780-2742-11e9-9a16-c0fab98bbd1b.jpeg)

 **Used the following architecture:** <br/>
 
 Conv layer 1: 64 filters with size of 7x7. <br/>
 Max Pooling layer: filter size 2x2. <br/>
 BatchNorm layer. <br/>
 Dropout with Keep prob : 0.85. <br/>
 
 Conv layer 2: 128 filters with size of 5x5. <br/>
 Max Pooling layer: filter size 2x2. <br/>
 BatchNorm layer. <br/>
 Dropout with Keep prob : 0.85. <br/>
 
 Conv layer 3: 256 filters with size of 3x3. <br/>
 Max Pooling layer: filter size 2x2. <br/>
 BatchNorm layer. <br/>
 Dropout with Keep prob : 0.85. <br/>
 
 Conv layer 4: 512 filters with size of 3x3. <br/>
 Max Pooling layer: filter size 2x2. <br/>
 BatchNorm layer. <br/>
 Dropout with Keep prob : 0.85. <br/>
 
 Fully connected layer 1: 2048 hidden neurons <br/>
 Dropout with Keep prob : 0.5. <br/>
 Fully connected layer 2: 2048 hidden neurons <br/>
 Dropout with Keep prob : 0.5. <br/>
 Fully connected layer 3: 2048 hidden neurons <br/>
 Dropout with Keep prob : 0.5. <br/>

 **Additional details:** <br/>
 Used Adam Optimizer <br/>
 Used learning rate decay <br/>
 Used mini batch of size 40 <br/>
 Used xavier weight initialization <br/>
 Used relu activation in hidden states <br/>
 Used softmax in output layer <br/>
 Used cross entropy as a loss function <br/>
 Used early stopping <br/>
 
# Results:
- Overall training accuracy: 99.6%
- Overall training loss: 0.0120

![training_acc](https://user-images.githubusercontent.com/6074821/52183044-a9807f00-280c-11e9-8ac8-523fc344b017.png) ![training_loss](https://user-images.githubusercontent.com/6074821/52183059-cddc5b80-280c-11e9-8f3e-6e9f52b5d0f5.png)

- Validation accuracy: 53.5%
- Validation loss: 2.1864

![valid_acc](https://user-images.githubusercontent.com/6074821/52183076-f7958280-280c-11e9-9c41-db77f01c370b.png)
![valid_loss](https://user-images.githubusercontent.com/6074821/52183078-067c3500-280d-11e9-8523-17bba2698fdc.png)

- Test accuracy: 56.8%
- Test loss: 2.0936

**Some predictions from test set:** <br/>
![predictions3](https://user-images.githubusercontent.com/6074821/52183101-44795900-280d-11e9-8c38-e884a1b82a57.png)

# Usage:
Here we will discuss how to run the project and what each file is responsible of:

### DownloadData.py:
This script will download the CIFAR-100 dataset.

### ExtractData.py:
This script will extract the CIFAR-100 dataset.

### PreprocessData.py:
This script will load the data, normalize it, shuffle it, take 2k images from test set as a dev set, and save it in a pickle file.

### TrainModel.py:
This script will begin training on the training data and output the results(including on test set after it finishes training) and save the accuracy and loss graphs in output_images folder, and save the graph info for tensorboard in graph_info folder, and save the model itself in saved_model

### TestModel.py:
This script will load the model saved in best_model folder(which gave the best accuracy overall) and run it on the test set and output the results.

### Prediction_Interface.py:
This script will open a gui view for you to load an image and classify it using the model in best_model folder. <br/>

![untitled](https://user-images.githubusercontent.com/6074821/52183394-2d883600-2810-11e9-8164-c57fa0c7867e.jpg)

# Future work:

- Try different architectures (resnets, inception) 
- Try hierarchical softmax since the labels of CIFAR come in 2 categories (soft label,hard label) 

# Environment Used:
- Python 3.6.1
- Tensorflow 1.9
