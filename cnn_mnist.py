# Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

#Using layers(an API to create a neural network).
#Code below

#Intro to Convolutional Neural Networks
#Convolutional Neural Networks are current state of the art model architecture for image classification
#CNN's apply filters to raw pixel data of an image to extract and learn higher level features
#The model can use these features for classification.
#CNN's contain three components.
    #Convolutional Layers
        #apply a series of convolution filters to the image
        #For each subregion, the layer performs mathematical operations to produce a single value for the output
        #Convlutional layers then apply a ReLU activation function to the output to introduce nonlinearities to the model
    #Pooling Layers
        #downsample the image data extracted by the convolutional layers to reduce dimensionality
        #Decreases processing time.
        #Common pooling algorithm is max pooling whcih extracts 2 by 2 subregions of the feature map and keeps their maximum value and discards other values
    #Dense (fully connected) layers
        #Perform classification on the features extracted by the convolution layers and downsample by pooling layers
        #In this layer, every node is connected to every other node in the preceeding layer
#Typically, CNN's consist of a stack of convolutional nodes that perform feature extraction
#Each module is a convolutional layer followed by a pooling layer
#The last convolutional layer is followed by one or more dense layers that perform classification
#The final layer dense layer in a CNN contains a single node for each target class in the model

#Our CNN MNIST Classifier Architecture

#1. Convolutional Layer #1 applies 32 5*5 filters extracting 5*5 pixel subregions with ReLU activation functions
#2. Pooling Layer #1 Performs max pooling with a 2*2 filter and stride of 2(which specifies that pooled regions don't overlap)
#3. Convolutional Layer #2 Applies 64 5*5 filters, with ReLU activation function
#4. Pooling Layer #2 Performs max pooling again with 2*2 filter and stride 2
#5. Dense Layer #1 1,024 neurons, with dropout regularization rate of 0.4 to prevent overfitting(probability of 0.4 that any given element will be dropped during training)
#6. Dense Layer #2(Logits Layer): 10 neurons, one for each digit target class (0-9).

#tf.layers API has methods to create each of these layers
    #method conv2d()
        #constructs a two-dimensional convolutional layer.
        #takes a number of filters, filter kernel size, padding, and activation function as arguments
    #method max_pooling2d()
        #constructs a two-dimensional pooling layer using max-pooling algorithm
        #takes pooling filter size and stride as arguments
    #dense
        #constructs a dense layer
        #takes a number of neurons and activation function as arguments

            #all these methods take in a tensor as input and return a transformed tensor
            #to connect layers, just take the output of one layer-creation method and supply it as input to another

#application logic below

#input layer
def cnn_model_fn(features, labels, mode):
    
#methods in layer API for creating convolutional and pooling layers for 2d image data expect
#input tensors to have a shape of [batch_size, image_width, image_height, channels]
    #batch_size
        #size of the subset of examples to use when performing gradient descent during training
    #image_width
        #width of the example images
    #image_height
        #height of the example images
    #channels
        #number of color channels in the example images. For color images, the number of channels is 3(red, green, blue)
        #for monochrome images, there is just 1 channel(black)

#our MNIST data is composed of monochrome 28*28 pixel images, so the desired shape
#for our input layer is [batch_size, 28, 28, 1]

#to convert our input feature map(features) to this shape, we can perform the following reshape operation

    input_layer = tf.reshape(features, [-1, 28, 28, 1])
#the -1, for the batch size specifies that this dimension should be dynamically computed based on the number of input values in features, holding all other dimensions constant
#this treats batch size as a hyper parameter that we can tune
#for example if we feed examples into our model in batches of 5, features will contain 3,3920 values(one value for each pixel in each image), and the input layer will have a shape of [5, 28, 28, 1].
#if we feed in images in batches of 100, features will contain 78,400(28*28*100) total pixels(values) and input_layer will have a shape of [100, 28, 28, 1]

#Convolutional Layer #1
#We want to apply 32 5*5 filters to the input layer, with a ReLU activation function. We can use conv2d() method in the layers API to create this layer
    conv1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=32, #specifies the number of filters to apply, here it is 32
    kernel_size=[5, 5], #specifies the dimensions of the filters as [width, height] here [5, 5]
    padding="same", #accepts one of 2 enumerated values, either "valid" or "same" to specify that the output should have the same width and height values as the input tensor.
                    #we set padding="same" here which instructs tensorflow to add 0 values to the edges of the output tensor to preserve width and height of 28(without padding a 5*5 convolution over a 28*28 tensor will produce a 24*24 grid(24*24 locations to extract a 5*5 tile from). 
    activation=tf.nn.relu)  #specifies the activation function to apply to the output of the convolution. Here it is ReLU activation with tf.nn.relu

#The inputs argument specifies our input tensor, which must have the shape [batch_size, image_width, image_height, channels].
#Here we connect the first convolutional layer to the input_layer, which has the shape, [batch_size, 28, 28, 1]
#Note, conv2d() will instead accept [channels, batch_size, image_width, image_height] when passed data_format=channels_first.
#our output tensor produced has a shape of [batch_size, 28, 28, 32] the same width and height as the input, but now with 32 channels holding the output from each of the filters.

#Pooling Layer 1
#we need to connect our first pooling layer to our convolutional layer we just created.
#we can use the max_pooling2d() method in layers to construct a layer that performs max pooling with a 2*2 filter and stride of 2
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
#inputs specify the input tensor, with shape [batch_size, image_width, image_height, channels]
#Here, our input tensor is conv1, the output from the first convolutional layer, which has a shape of [batch_size, 28, 28, 32]
#Note, As with conv2d(), max_pooling2d() will instead accept a shape of [channels, batch_size, image_width, image_height] when passed the argument data_format=channels_first.
#pool size argument specifies the size of the max pooling filter as [width, height], here it is [2, 2]
#strides argument specifies the size of the stride. Here it is 2 which indicates that the subregions extracted by the filter should be seperated by 2 pixels in both the width and height dimensions. This means that none of the regions will overlap.
#Our output vector from max_pooling2d() will have a shape of [batch_size, 14, 14, 32] the 2*2 filter reduces the width and height by 50% each

#Convolutional Layer #2
#construct a 2nd convolutional and pooling layer to our CNN using conv2d() and max_pooling2d() like before.
#For convolutional layer 2 we configure 64, 5*5 filters with ReLU activation and for pooling layer 2 we use the same specs as pooling layer #1 (2*2 max pooling with stride 2)
    conv2 = tf.layers.conv2d(
    inputs=pool1,   #takes the output of our pooling layer 1, and produces tensor h_conv2 as output
    filters=64, 
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)
#conv2 has shape [batch_size, 14, 14, 64] the same width and height as pool 1 due to padding="same" and 64 channels for the 64 filters applied

#Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
#takes conv2 as input producing pool2 as output.
#pool 2 has shape [batch_size, 7, 7, 64] (50% reduction of width and height from conv2)

#Dense Layer
#We want to add a dense layer with 1,024 neurons and ReLU activation to our CNN to perform classification on the features extracted by the convolution/pooling layers. Before we connecct the layer, however, we'll flatten our feature map(pool2) to shape [batch_size, features] so that our tensor has only 2 dimensions
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
#in reshape operation, -1 signifies that the batch_size dimensions will be dynamically calculated based on the number of examples in our input data.
#Each example has 7(pool2 width) * 7(pool2 height) * 64(pool2 channels) features so we want the features dimension to have a value of 7*7*64 = 3136 in total
#The output tensor, pool2_flat has a shape [batch_size, 3136]

#Now we use the dense method in layers to connect our dense layer as follows
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
#the inputs argument specifies the input tensor: our flattened feature map(pool2_flat).
#The units argument specifies the number of neurons in the dense layer(1024)
#The activation argument takes the activation function, again we'll use tf.nn.relu to add ReLU activation

#To improve the results of our model, we'll also apply regularization to our dense layer, using the dropout method in layers:
    dropout = tf.layers.dropout(
    inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
#inputs specifies the input tensor which is the output tensor from our dense layer(dense)
#The rate argument specifies the dropout rate; here we use 0.4 which means 40% of the elements will be randomly dropped out during training.
#The training argument takes both a boolean specifying wether or not the model is currently being run in training mode, dropout will only be performed if training is True.
#Here we check if the mode passed to our model function cnn_model_fn is TRAIN mode.
#Our output tensor dropout has shape [batch_size, 1024]

#Logits Layer
#The final layer in our neural net is the logits layer, which will return the raw values for our predications.
#We create a dense layer with 10 neurons(one for each target class 0-9), with linear activation(the default)
    logits = tf.layers.dense(inputs=dropout, units=10)
#Our final output tensor of the cnn, logits has shape[batch_size, 10]

#Calculating Loss
#Use cross entropy for this multiclassification problem
    loss = None
    train_op = None

# Calculate loss for both TRAIN and EVAL modes
    if mode != learn.ModeKeys.INFER:
      onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
      loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)
#label holds our predictions for the examples ie [1, 2, 5, 7, ..... ] we next have to convert our predicted labels to one-hot encoding we can use tf.one_hot to accomplish this conversion
#tf.one_hot() has 2 required arguments
    #indices
        #The locations in the one-hot tensor that will have "on values" ie the locations of 1 values in the tensor
    #depth
          #the depth of the one hot tensor- ie the number of target classes. Here, the depth is 10

#The following code creates the one-hot tensor for our labels, onehot_labels
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
#labels contains a series from 0-9 indices is just our labels tensor with the values cast to integers.
#the depth is 10 because we have 10 possible target classes, one for each digit.

#Next we compute cross entropy of onehot_labels and the softmax of the predictions from our logits layer
#tf.losses.softmax_cross_entropy() takes onehot_labels and logits as args performs softmax activation on logits, calculates cross-entropy and returns our loss as a scalar Tensor
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

#configuring the training OP
#configure our model to optimize this loss value during training, using tf.contrib.layers.optimize_loss method in tf.contrib.layers
#We'll use a learning rate of 0.001 and stochastic gradient descent as the optimizing algorithm
# Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="SGD")

#Generate predictions
#The logits layer of our model returns our predications as raw values in a [batch_size, 10] - dimensional tensor. Let's convert these raw values into different formats that our model function can return.
    #The predicted class for each example: a digit from 0-9
    #the probabilities for each possible target class for each example. The probability that the example is a 0, is a 1, is a 2, etc.
#For a given example, our predicted class is the element in the corresponding row of the logits tensor with the highest raw value.
#we can find the index of this element using tf.argmax function.
    tf.argmax(input=logits, axis=1)
#The input arg specifies the tensor from which to extract the max value(in this case logits). The axis arg specifies the axis of the input tensor along which to find the greatest value. Here we want to find the largest value along the dimension with index of 1, which correpsonds to our predictions
#our logits tensor has shape [batch_size, 10]

#We can derive the probabilities from our logits layer by applying softmax activation using tf.nn.softmax
    tf.nn.softmax(logits, name="softmax_tensor")
#We use the name argument to explicitly name this operation softmax_tensor, so we can reference it later. (We'll set up logging for the softmax values in "Set Up a Logging Hook".

#We compile the predictions in a dict as follows:
    predictions = {
    "classes": tf.argmax(
        input=logits, axis=1),
    "probabilities": tf.nn.softmax(
        logits, name="softmax_tensor")
}

#finally we can return our prediction, loss, and train_op in a tf.contrib.learn.ModelFnOps object:
# Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op)

#Main function to load the data
def main(unused_argv):
  # Load training and eval data
  mnist = learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
  #Creating the estimator
  mnist_classifier = learn.Estimator(
model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
  #model_fn arg specifies the model function to use for training, evaluation, and inferences; we pass it the cnn_model_fn we created
  #the model_dir argument specifies the directory where the model data(checkpoints) will be saved.

  #since CNN's take a while to train, we are going to set up logging so we can track progress.
  #use tensorflow's tf.train.SessionRunHook to create a tf.train.LoggingTensorHook that will log the probability values from the softmax layer of our CNN. 
  #Set up logging for predictions
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)
  #store a dict of tensors we want to log in tensor_to_log
  #set every_n_iter to 50 which specifies that probabilities should be logged after every 50 steps of training
  #training the model by calling fit on mnist_classifier
  mnist_classifier.fit(
    x=train_data,
    y=train_labels,
    batch_size=100, #model will train on mini batches of 100
    steps=2000,#model will train for 20,000 steps
    monitors=[logging_hook]) #so we can get feedback during training

   # Configure the accuracy metric for evaluation
  metrics = {
      "accuracy":
          learn.MetricSpec(
              metric_fn=tf.metrics.accuracy, prediction_key="classes"),
  }

    # Evaluate the model and print results
  eval_results = mnist_classifier.evaluate(
        x=eval_data, y=eval_labels, metrics=metrics)
  print(eval_results)
#store the training feature data(raw pixel values for 55,000 images of hand drawn digits)
#and training labels(the corresponding values from 0-9 for each image)
#as numpy array in train_data and train_labels respectively.
#Similarly, we store the evaluation feature data(10,000 images) and evaluation labels in eval_data and eval_labels, respectively.

if __name__ == "__main__":
  tf.app.run()

