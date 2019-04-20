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
- 10k as a test set

# Preprocessing:
- Used [Unsharp masking kernel](https://en.wikipedia.org/wiki/Unsharp_masking) to sharpen the blurry images 

![image](https://user-images.githubusercontent.com/6074821/53117205-0ee5a700-3553-11e9-969c-e5bc84c2299b.png)

- Normalized pixel values to be between -1 and 1 (done during training)
- Used data augmentation on training set (flip horizontally, crop from edges) (done randomly per training batch)


# Training Method:

We tried different models and techniques like normal CNNs, Depth-wise separable convolution, and resnets. Resnets provided the best results

![image](https://user-images.githubusercontent.com/6074821/56460821-27ced500-63a9-11e9-8e3d-444af6d725ad.png)

 **Used the following architecture:** <br/>
 
 Residual block 1: 128 filters with size of 3x3. <br/>
 Dropblock: filter size 5x5. <br/>
 Maxpool layer with size 3x3. <br/>
 
 Residual block 2: 256 filters with size of 3x3. <br/>
 Dropblock: filter size 5x5. <br/>
 Maxpool layer with size 3x3. <br/>
 
 Residual block 3: 512 filters with size of 3x3. <br/>
 Dropblock: filter size 5x5. <br/>
 Maxpool layer with size 3x3. <br/>
 
 Residual block 4: 1024 filters with size of 3x3. <br/>
 Dropblock: filter size 5x5. <br/>
 Maxpool layer with size 3x3. <br/>
 
 Fully connected layer 1: 4096 hidden neurons <br/>
 Dropout with Keep prob : 0.5. <br/>
 Fully connected layer 2: 4096 hidden neurons <br/>
 Dropout with Keep prob : 0.5. <br/>
 Fully connected layer 3: 4096 hidden neurons <br/>
 Dropout with Keep prob : 0.5. <br/>
 
 With each residual block consisting of:<br/>
 Batch norm. <br/>
 Convolution. <br/>
 Batch norm. <br/>
 Convolution. <br/>
 

 **Additional details:** <br/>
 Used Adam Optimizer <br/>
 Used learning rate decay <br/>
 Used mini batch of size 250 <br/>
 Used xavier weight initialization <br/>
 Used relu activation in hidden states <br/>
 Used softmax in output layer <br/>
 Used cross entropy as a loss function <br/>
 Used early stopping <br/>
 
# Results:
- Overall training accuracy: 99.5%
- Overall training loss: 0.0300

![image](https://user-images.githubusercontent.com/6074821/56460395-50070580-63a2-11e9-935f-3fb869fd0f90.png)
![image](https://user-images.githubusercontent.com/6074821/56460400-5ac19a80-63a2-11e9-9aae-d717f55dbf84.png)

- Test accuracy: 70.7%
- Test loss:  1.2815

![image](https://user-images.githubusercontent.com/6074821/56460401-63b26c00-63a2-11e9-9576-5fdf1491a46e.png)
![image](https://user-images.githubusercontent.com/6074821/56460402-6b721080-63a2-11e9-9367-81e6e5eb1e21.png)

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

### TrainCNN.py:
This script will begin training the vanilla CNN model on the training data, output the results, save the accuracy and loss graphs in output_images folder, save the graph info for tensorboard in graph_info folder, and save the model itself in saved_model. You can expect around 65% accuracy on test set.

### TrainDepthwiseCNN.py:
This script will begin training the faster depth-wise CNN model on the training data, output the results, save the accuracy and loss graphs in output_images folder, save the graph info for tensorboard in graph_info folder, and save the model itself in saved_model. You can expect around 60% accuracy on test set.

### TrainResNet.py:
This script will begin training the resnet model described above on the training data, output the results, save the accuracy and loss graphs in output_images folder, save the graph info for tensorboard in graph_info folder, and save the model itself in saved_model. You can expect around 70% accuracy on test set.

### TestModel.py:
This script will load the model saved in best_model folder(which gave the best accuracy overall) and run it on the test set and output the results.

### Prediction_Interface.py:
This script will open a gui view for you to load an image and classify it using the model in best_model folder. <br/>

![untitled](https://user-images.githubusercontent.com/6074821/52183394-2d883600-2810-11e9-8164-c57fa0c7867e.jpg)

# Future work:
- Try different architectures 
- Try hierarchical softmax since the labels of CIFAR come in 2 categories (soft label, hard label) 

# Environment Used:
- Python 3.6.1
- Tensorflow 1.10
- imgaug 0.2.8
- opencv-python 4.0.0
