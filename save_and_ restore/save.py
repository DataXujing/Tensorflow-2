import tensorflow as tf
import pandas as pd
import numpy as np
import time
#载入数据集
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.zeros(shape)+0.1
    return tf.Variable(initial)

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

train = train.sample(frac=1).reset_index(drop=True)

feature_name = train.columns[1:] 
label_name = train.columns[0] #put the class label on the first column

train_feature = train[feature_name].as_matrix()
test_feature = test[feature_name].as_matrix()

labels = sorted(set(train[label_name].tolist()))

train_label_one_hot = array_to_one_hot(train[label_name].tolist(),labels) #tensorflow must use one-hot form
test_label_one_hot = array_to_one_hot(test[label_name].tolist(),labels)

train_label_array = np.array(train[label_name])
test_label_array = np.array(test[label_name])

# imput image is 28*28
n_inputs = 28
max_time = 28 #28 lines in total
lstm_size = 100
n_classes = 10
batch_size = 100
learning_period = 10
learning_rate = 5*1e-4 
n_batch = len(train) // batch_size #计算一共有多少个批次

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

# initial weight and bias
weights = weight_variable([lstm_size, n_classes])
biases = bias_variable([n_classes])
#定义RNN网络
# inputs=[batch_size, max_time, n_inputs]
inputs = tf.reshape(x,[-1,max_time,n_inputs])
#定义LSTM基本CELL
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
# final_state[0]是cell state
# final_state[1]是hidden_state
outputs,final_state = tf.nn.dynamic_rnn(lstm_cell,inputs,dtype=tf.float32)
prediction = tf.nn.softmax(tf.matmul(final_state[1],weights) + biases)    
    
#计算RNN的返回结果
#损失函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
#使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
#结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax返回一维张量中最大的值所在的位置
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))#把correct_prediction变为float32类型
#初始化
time_0 = time.time()
saver = tf.train.Saver()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(learning_period):
        for k in range(n_batch):
            batch_x = next_batch(k,batch_size,train_feature)
            batch_y = next_batch(k,batch_size,train_label_one_hot)
            sess.run(train_step,feed_dict={x:batch_x,y:batch_y})
        
        train_acc = sess.run(accuracy,feed_dict={x:train_feature,y:train_label_one_hot})
        test_acc = sess.run(accuracy,feed_dict={x:test_feature,y:test_label_one_hot})
        print ("period %d"%epoch,"train:",train_acc, "test:",test_acc)
    saver.save(sess,'NET/RNN_net.ckpt')
print("total time:", time.time()-time_0)