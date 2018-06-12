import csv
import numpy as np
from .data import DataBody
import tensorflow as tf

class InputBody():

	###############
	def __init__(self, experiment_name = 'default',mode = 'bodyfat',
				 batch_size = '10'):
		self.experiment_name = experiment_name
		data_body = DataBody(experiment_name = self.experiment_name)
		
		if mode == 'bodyfat':
			self.train_out = data_body.data_train[:,1]
			self.test_out = data_body.data_test[:,1]
		elif mode == 'Density':
			self.train_out = data_body.data_train[:,0]
			self.test_out = data_body.data_test[:,0]
			
		self.train_in = data_body.data_train[:,2:-1]
		self.test_in = data_body.data_test[:,2:-1]
		
		self.batch_size = batch_size
			
	###############
	def placeholder_inputs(self, batch_size = self.batch_size):
	  """Generate placeholder variables to represent the input tensors.
	  These placeholders are used as inputs by the rest of the model building
	  code and will be fed from the downloaded data in the .run() loop, below.
	  Args:
		batch_size: The batch size will be baked into both placeholders.
	  Returns:
		images_placeholder: Images placeholder.
		labels_placeholder: Labels placeholder.
	  """
	  # Note that the shapes of the placeholders match the shapes of the full
	  # image and label tensors, except the first dimension is now batch_size
	  # rather than the full size of the train or test data sets.
	  
	  input_placeholder = tf.placeholder(self.train_in.dtype, 
	  									 shape=(batch_size,
	  									 self.train_in.shape[1])
	  								
	  output_placeholder = tf.placeholder(self.train_out.dtype, 
	  									 shape=(batch_size,
	  									 self.train_out.shape[1])

	  return input_placeholder, output_placeholder
	  
'''  
	  with np.load("/var/data/training_data.npy") as data:
	  
	  
  features = data["features"]
  labels = data["labels"]

# Assume that each row of `features` corresponds to the same row as `labels`.
assert features.shape[0] == labels.shape[0]

features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
# [Other transformations on `dataset`...]
dataset = ...
iterator = dataset.make_initializable_iterator()
'''
sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                          labels_placeholder: labels})
