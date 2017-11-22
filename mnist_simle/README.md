# Mnist_tensorflow
tensorflow example, using minist data

I first pull the minist data as csv file. So you can easily change to your own data to perform classification!

The minist_get_train_test.py is to get csv form of the data.
For train, I use 1000 each for 0-9, for test, I use 200 each for 0-9.
I change the original label 0-9 to 10-19. It is easier for you to distinguish the predicted index and label.

The mnist_2_layer.py is a sample of two hidden layers and for each layer I use relu activation function.
For the loss, I use softmax_cross_entropy_loss.

I output predicted probability matrix, predicted label, train and test accuarcy.

