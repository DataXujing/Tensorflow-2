import tensorflow as tf
import os
import tarfile
import requests

#download address
inception_pretrain_model_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

#save the model
inception_pretrain_model_dir = "inception_model"
if not os.path.exists(inception_pretrain_model_dir):
    os.makedirs(inception_pretrain_model_dir)
    
#get file name, file path
file_name = inception_pretrain_model_url.split('/')[-1]
file_path = os.path.join(inception_pretrain_model_dir, file_name)

print("file name:",file_name)
print("file path:",file_path)

#download model
if not os.path.exists(file_path):
    print("download: ", file_name)
    r = requests.get(inception_pretrain_model_url, stream=True)
    with open(file_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
print("finish downlad")

#unzip the model
tarfile.open(file_path, 'r:gz').extractall(inception_pretrain_model_dir)

# save log in order to visulaize in tensorboard
log_dir = 'inception_log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
#classify_image_graph_def.pb is the trained model
inception_graph_def_file = os.path.join(inception_pretrain_model_dir, 'classify_image_graph_def.pb')
with tf.Session() as sess:
    with tf.gfile.FastGFile(inception_graph_def_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    #save the log
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    writer.close()
