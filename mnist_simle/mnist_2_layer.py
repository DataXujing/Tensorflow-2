import tensorflow as tf
import pandas as pd
import numpy as np

def array_to_one_hot(lable):
    # input lable is list
    new_lable =  []
    lables =  sorted(set(lable))    
    for i in lable:
        single_lable = [0] * len(lables)
        index = lables.index(i)
        single_lable[index] = 1
        new_lable.append(single_lable)
    return np.array(new_lable)

def fake_label_to_real_label_array(fake_label, labels):
    real_label_array = []
    for i in fake_label:
        real_label_array.append(labels[i])
    return np.array(real_label_array)

def one_hot_to_array(labels):
    new_label = []
    for label in labels:
        index = label.tolist().index(1)
        new_label.append(index)
    return new_label

train = pd.read_csv("mnist_train.csv")
test = pd.read_csv("mnist_test.csv")

feature_name = train.columns[1:] 
label_name = train.columns[0] #put the class label on the first column

train_feature = train[feature_name].as_matrix()
test_feature = test[feature_name].as_matrix()

train_label_one_hot = array_to_one_hot(train[label_name].tolist()) #tensorflow must use one-hot form
test_label_one_hot = array_to_one_hot(test[label_name].tolist())

train_label_array = np.array(train[label_name])
test_label_array = np.array(test[label_name])

labels = sorted(set(train[label_name].tolist()))

## compelet of preparing the data
learning_rate = 0.005
training_period = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 400 # 1st layer number of features
n_hidden_2 = 200 # 2nd layer number of features
n_input = train_feature.shape[1]
n_classes = train_label_one_hot.shape[1]

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# define variable
w1 = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
w2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
w3 = tf.Variable(tf.random_normal([n_hidden_2, n_classes]))

b1 = tf.Variable(tf.random_normal([n_hidden_1]))
b2 = tf.Variable(tf.random_normal([n_hidden_2]))
b3 = tf.Variable(tf.random_normal([n_classes]))

layer_1 = tf.add(tf.matmul(x, w1), b1)
layer_1 = tf.nn.relu(layer_1)
# Hidden layer with RELU activation
layer_2 = tf.add(tf.matmul(layer_1, w2), b2)
layer_2 = tf.nn.relu(layer_2)
# Output layer with linear activation
pred = tf.add(tf.matmul(layer_2, w3), b3)
    
# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

def next_batch(round_num,sample_num, data):
    # Return a total of `num` samples from the array `data`. 
    return data[sample_num*round_num:sample_num*(round_num+1)]

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for k in range(training_period):
        total_batch = int(train_feature.shape[0]/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x = next_batch(i,batch_size,train_feature)
            batch_y = next_batch(i,batch_size,train_label_one_hot)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, loss], feed_dict={x: batch_x,
                                                          y: batch_y})
        # Display logs per epoch step
        if k % display_step == 0:
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            
            test_accuracy = accuracy.eval({x: test_feature, y: test_label_one_hot})
            train_accuarcy = accuracy.eval({x: train_feature, y: train_label_one_hot})
            print("period %d" %k,"trian:", train_accuarcy, "test:" ,test_accuracy)
    
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # get accuarcy
    test_accuracy = accuracy.eval({x: test_feature, y: test_label_one_hot})
    train_accuarcy = accuracy.eval({x: train_feature, y: train_label_one_hot})
    
    # get probability matrix
    soft_y = tf.nn.softmax(pred)
    test_pred_prob_matrix = soft_y.eval(feed_dict={x: test_feature})
    train_pred_prob_matrix = soft_y.eval(feed_dict={x: train_feature})
    
    # get predicted index
    test_pred_index = sess.run(tf.argmax(pred, 1), feed_dict={x: test_feature})
    train_pred_index = sess.run(tf.argmax(pred, 1), feed_dict={x: train_feature})
    
    # get predicted label
    test_pred_label_array = fake_label_to_real_label_array(np.array(test_pred_index.tolist()), labels) #list form
    train_pred_label_array = fake_label_to_real_label_array(np.array(train_pred_index.tolist()), labels) #list form 