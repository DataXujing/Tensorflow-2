import tensorflow as tf
import numpy as np
import pandas as pd

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.zeros(shape)+0.1
    return tf.Variable(initial)

def get_conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool(x,ksize):
    return tf.nn.max_pool(x,ksize,strides=[1,2,2,1],padding='SAME')

def next_batch(round_num,sample_num,data):
    total_batch = len(data)//sample_num
    round_num = round_num % total_batch
    return data[sample_num*round_num:sample_num*(round_num+1)]

def array_to_one_hot(all_lables,lables):
    # input lable is list
    new_lable =  [] 
    for i in all_lables:
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

train = pd.read_csv("mnist_train.csv")
test = pd.read_csv("mnist_test.csv")

train = train.sample(frac=1)

feature_name = train.columns[1:] 
label_name = train.columns[0] #put the class label on the first column

train_feature = train[feature_name].as_matrix()
test_feature = test[feature_name].as_matrix()

labels = sorted(set(train[label_name].tolist()))

train_label_one_hot = array_to_one_hot(train[label_name].tolist(),labels) #tensorflow must use one-hot form
test_label_one_hot = array_to_one_hot(test[label_name].tolist(),labels)

train_label_array = np.array(train[label_name])
test_label_array = np.array(test[label_name])


learning_rate_start = 0.0001
training_period = 1501
batch_size = 100
display_step = 100

image_reshape = [-1,28,28,1]
n_input = train_feature.shape[1]
n_classes = train_label_one_hot.shape[1]
conv1_stru = [5,5,1,32]
conv2_stru = [5,5,32,64]
pool1_ksize = [1,2,2,1]
pool2_ksize = [1,2,2,1]
pool2_flat = [-1,7*7*64]
fc1_stru = [7*7*64,1024]
fc2_stru = [1024,10]


x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(x,image_reshape)

w_conv1 = weight_variable(conv1_stru)
w_conv2 = weight_variable(conv2_stru)
w_fc1 = weight_variable(fc1_stru)
w_fc2 = weight_variable(fc2_stru)  

b_conv1 = bias_variable([conv1_stru[-1]])
b_conv2 = bias_variable(conv2_stru[-1])
b_fc1 = bias_variable([fc1_stru[-1]])
b_fc2 = bias_variable([fc2_stru[-1]])

# convolution layer 1
conv_1 = get_conv2d(x_image,w_conv1) + b_conv1
conv_1 = tf.nn.relu(conv_1)
# maxpooling layer 1
pool_1 = max_pool(conv_1,pool1_ksize)

# convolution layer 2
conv_2 = get_conv2d(pool_1,w_conv2) + b_conv2
conv_2 = tf.nn.relu(conv_2)

pool_2 = max_pool(conv_2,pool2_ksize)

pool_2_flat = tf.reshape(pool_2,pool2_flat)

fc_1 = tf.matmul(pool_2_flat,w_fc1) + b_fc1
fc_1_act = tf.nn.relu(fc_1)
fc_1_drop = tf.nn.dropout(fc_1_act,keep_prob)

fc_2 = tf.matmul(fc_1_drop,w_fc2) + b_fc2
fc_2_act = tf.nn.relu(fc_2)
fc_2_drop = tf.nn.dropout(fc_2_act,keep_prob)

pred = tf.nn.softmax(fc_2_drop)
batch = tf.Variable(0)
learning_rate = tf.train.exponential_decay(learning_rate_start,
                                            training_period,
                                            10000,
                                            0.97,
                                            staircase=True)

loss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss,global_step=batch)


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for k in range(training_period):
        #print("k is:",k)
        total_batch = int(train_feature.shape[0]/batch_size)
        # Loop over all batches
        batch_x = next_batch(k,batch_size,train_feature)
        batch_y = next_batch(k,batch_size,train_label_one_hot)
        # Run optimization op (backprop) and cost op (to get loss value)
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y,keep_prob:0.9})
        # Display logs per epoch step
        if k % display_step == 0:
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            
            test_accuracy = accuracy.eval({x: test_feature, y: test_label_one_hot,keep_prob:1})
            train_accuarcy = accuracy.eval({x: train_feature, y: train_label_one_hot,keep_prob:1})
            print("period %d" %k,"trian:", train_accuarcy, "test:" ,test_accuracy)
    
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # get accuarcy
    test_accuracy = accuracy.eval({x: test_feature, y: test_label_one_hot,keep_prob:1})
    train_accuarcy = accuracy.eval({x: train_feature, y: train_label_one_hot,keep_prob:1})
    
    test_acc = sess.run(accuracy,feed_dict={x: test_feature, y: test_label_one_hot,keep_prob:1.0})
    train_acc = sess.run(accuracy,feed_dict={x: test_feature, y: test_label_one_hot,keep_prob:1.0})
    # get probability matrix
    soft_y = tf.nn.softmax(pred)
    test_pred_prob_matrix = sess.run(soft_y,feed_dict={x: test_feature,keep_prob:1.0})
    train_pred_prob_matrix = sess.run(soft_y,feed_dict={x: train_feature,keep_prob:1.0})
    
    # get predicted index
    test_pred_index = sess.run(tf.argmax(pred, 1), feed_dict={x: test_feature,keep_prob:1.0})
    train_pred_index = sess.run(tf.argmax(pred, 1), feed_dict={x: train_feature,keep_prob:1.0})
    
    # get predicted label
    test_pred_label_array = fake_label_to_real_label_array(np.array(test_pred_index.tolist()), labels) #list form
    train_pred_label_array = fake_label_to_real_label_array(np.array(train_pred_index.tolist()), labels) #list form 

    
