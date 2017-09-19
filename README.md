# Tensorflow_more

I add more things on tensorflow, such as dropout and use tensorboard to visualization.

## Tensorflow Dropout
I add a dropout probality for hidden layer. This method may help to speed up convergence.

## Tensorflow CNN
I still use the mnist data.

+ input layer : 784 nodes (MNIST images size)
+ first convolution layer : 5x5x32
+ first max-pooling layer
+ second convolution layer : 5x5x64
+ second max-pooling layer
+ third fully-connected layer : 1024 nodes
+ output layer : 10 nodes (number of class for MNIST)
+ for learning rate, I use 0.0001
