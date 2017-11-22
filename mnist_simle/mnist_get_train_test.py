import pandas as pd
import numpy as np
from pandas import DataFrame
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

def one_hot_to_array(labels):
    new_label = []
    for label in labels:
        index = label.tolist().index(1)
        new_label.append(index)
    return new_label

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_feature = DataFrame(mnist.train.images)
test_feature = DataFrame(mnist.test.images)

train_label = DataFrame(one_hot_to_array(mnist.train.labels))+10
test_label = DataFrame(one_hot_to_array(mnist.test.labels))+10

train_data = pd.concat([train_label,train_feature],axis = 1)
test_data = pd.concat([test_label,test_feature],axis = 1)

column_name = ['class']
for i in range(train_feature.shape[1]):
    column_name.append('x_{}'.format(i))

train_data.columns = column_name
test_data.columns = column_name

print (train_data['class'].value_counts())
print (test_data['class'].value_counts())


## get 10000 train data and 2000 test data
new_train_data = DataFrame()
new_test_data = DataFrame()

for i in range(10,20):
    train_temp = train_data.loc[train_data['class'] == i].sample(frac=1).reset_index(drop=True)
    test_temp = test_data.loc[test_data['class'] == i].sample(frac=1).reset_index(drop=True) 
    new_train_data = new_train_data.append(train_temp[:1000])
    new_test_data = new_test_data.append(train_temp[:200])

print (new_train_data['class'].value_counts())
print (new_test_data['class'].value_counts())

## save data to csv
new_train_data.to_csv("mnist_train.csv",index = False)
new_test_data.to_csv("mnist_test.csv",index = False)