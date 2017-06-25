#Using layers(an API to create a neural network).

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


#Our 
        

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here

if __name__ == "__main__":
  tf.app.run()

