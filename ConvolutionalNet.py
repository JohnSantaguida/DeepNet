import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import prettytensor as pt

import NetworkHelper as nh

#LOAD (MNIST---) DATA
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=True)
#store outputs as nums not arrays
data.test.cls = np.argmax(data.test.labels, axis=1)

'''
#see set sizes
print("Size of: ")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))
#see 1hot and regular labels
print (data.test.labels[0:5, :])
print (data.test.cls[0:5])
#see version
print ("Tensorflow Version : " + tf.__version__)
'''

#Convolutional Layers Specs
filtersize1 = 5
numfilter1 = 16
filtersize2 = 5
numfilter2 = 36
connectedsize = 128

#Data dimensions
img_size=28
img_size_flat= img_size*img_size
img_shape = (img_size, img_size)
num_color_channels = 1
num_classes = 10

# Get the first images from the test-set.
images = data.test.images[0:9]

# Get the true classes for those images.
cls_true = data.test.cls[0:9]

# Plot the images and labels using our helper-function above.
#nh.plot_images(img_shape, images=images,  cls_true=cls_true)

#Placeholders
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_img = tf.reshape(x, [-1, img_size, img_size, num_color_channels])
y_true = tf.placeholder(tf.float32, shape=[None,10], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


#Create Layer 1
layer_conv1, weights_conv1 = \
    nh.new_conv_layer(input=x_img, num_input_channels=num_color_channels,
                      filter_size=filtersize1, num_filters=numfilter1, use_pooling=True)
#print (layer_conv1)
#Create Layer 2
layer_conv2, weights_conv2 = \
    nh.new_conv_layer(input=layer_conv1, num_input_channels=numfilter1,
                      filter_size=filtersize2, num_filters=numfilter2, use_pooling=True)
#print (layer_conv2)

#Make flat layer
layer_flat, num_features = nh.flatten_layer(layer_conv2)
#print (layer_flat)
#print (num_features)

#Make fully connected layer
layer_fc1 = nh.new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=connectedsize, use_relu=True)
#print(layer_fc1)
#Make fully connected2
layer_fc2 = nh.new_fc_layer(input=layer_fc1, num_inputs=connectedsize, num_outputs=num_classes, use_relu=False)
#print(layer_fc2)

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#TF
session = tf.Session()
session.run(tf.global_variables_initializer())

#Optimization helpr
train_batch_size = 64
total_iterations =0
def optimize(num_iterations):
    global total_iterations
    start_time=time.time()
    for i in range(total_iterations, total_iterations+num_iterations):
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        feed_dict_train = {x: x_batch, y_true: y_true_batch}
        session.run(optimizer,feed_dict=feed_dict_train)
        if i%100 == 0:
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            msg="Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i+1, acc))
    total_iterations+=num_iterations
    end_time = time.time()
    runtime = end_time-start_time
    print ("Time usage: " + str(timedelta(seconds=int(round(runtime)))))

def plot_example_errors(cls_pred, correct):
    incorrect = (correct==False)
    images = data.test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = data.test.cls[incorrect]
    nh.plot_images(img_shape, images=images[0:9], cls_true=cls_true[0:9],cls_pred=cls_pred[0:9])

def plot_confusion_matrix(cls_pred):
    cls_true = data.test.cls
    cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)
    print(cm)
    plt.matshow(cm)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

test_batch_size = 256
def print_test_accuracy(show_example_errors=False, show_confusion_matrix=False):
    num_test = len(data.test.images)
    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    i=0
    while i < num_test:
        j = min(i + test_batch_size, num_test)
        images = data.test.images[i:j, :]
        labels = data.test.labels[i:j, :]
        feed_dict = {x: images, y_true:labels}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i=j
    cls_true = data.test.cls
    correct = (cls_true==cls_pred)
    correctsum = correct.sum()
    acc = float(correctsum)/num_test
    msg = "Accuracy on test-set: {0:.1%} ({1}/{2})"
    print (msg.format(acc, correctsum, num_test))
    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)
    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

'''
print_test_accuracy()
optimize(num_iterations=1)
print_test_accuracy()
optimize(num_iterations=99)
print_test_accuracy()#show_example_errors=True)
optimize(num_iterations=900)
print_test_accuracy(show_example_errors=True)
optimize(num_iterations=9000)
'''
optimize(num_iterations=1000)
print_test_accuracy(show_example_errors=True, show_confusion_matrix=True)

#SAVING???
#saver = tf.train.saver()
#save_path = 'save/best_validation'

