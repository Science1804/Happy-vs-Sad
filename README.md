# Happy-vs-Sad
Convolution Neural Network for detecting Happy and Sad Emoticons.

Using Tensorflow library , an open source library for Numerical Computation for building a Convolution Neural Network for predicting happy-vs-sad emoticons. <br />
It just a simple code for understanding how to work with tensorflow .<br />
There will be a Git <br />
The following steps are followed:<br />
1)- Extracting the Zip-file.<br />
2)- Defining a Callback function (not neccesary always, Can be neglected if want - removing the further part will be mentioned)<br />
3)- Defining Model with Convolution and Max Pooling Layers (Number of layers and the neurons can be tweaked as indiviual depending on inviduals wish , But the number of Conv2D and MaxPooling2D must be same).<br />
4)- Compile model with RMSprop and loss function as 'binary_crossentropy' ,since we are classifying between happy and sad<br />
5)- Load ImageDataGenerator - it will help us to load the images from directory directly into the the model while training .( it can be tweaked as per requirement but we are doing it for a simple scenario).<br />
6)- form a train_generator using ImageDataGenerator where we will map the directory of training images.<br />
We are not forming a validation generator since it is a small data set and we are just learning here to make a simple CNN model.<br />
7)- fit the generator on  model and train it.<br />
8)- plot the Epochs vs Accuracy and Epochs vs Loss - using matplot.pyplot.<br />
