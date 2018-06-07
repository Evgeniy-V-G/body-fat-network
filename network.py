import csv
import numpy as np

def inference_network(inputs,input_size, hidden1_units, hidden2_units):
  """Build the MNIST model up to where it may be used for inference.
  Args:
    images: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.
  Returns:
    softmax_linear: Output tensor with the computed logits.
  """

  # Hidden 1
  with tf.name_scope('hidden1'):
  
  	# Random values to initialize
    weights = tf.Variable(
        tf.truncated_normal([input_size, hidden1_units],
                            stddev=1.0 / math.sqrt(float(input_size))),
        name='weights')
    
    # Constant inputs 
    biases = tf.Variable(tf.zeros([hidden1_units]),name='biases')
    
   	# Activation Function
    hidden1 = tf.nn.relu(tf.matmul(inputs, weights) + biases)
    
  # Hidden 2
  with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
        					name='weights')
        					
    biases = tf.Variable(tf.zeros([hidden2_units]),name='biases')
    
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    
  # Linear
  with tf.name_scope('output'):
    weights = tf.Variable(tf.truncated_normal([hidden2_units, 1],
                          stddev=1.0 / math.sqrt(float(hidden2_units))),
                          name='weights')

    biases = tf.Variable(tf.zeros([1]),name='biases')
    
    logits = tf.matmul(hidden2, weights) + biases

  return logits