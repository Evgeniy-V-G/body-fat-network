import csv
import numpy as np
from .network import BodyNetwork

def TrainerBody(self, input_body):

	###############
	def __init__(self,):
		self.hidden1_units = 10
		self.hidden2_units = 10
		self.input_body = input_body
		self.body_nework = BodyNetwork()
	
	
	def run_training():
	  """Train MNIST for a number of steps."""
	  # Get the sets of images and labels for training, validation, and
	  # test on MNIST.
	  data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)

	  # Tell TensorFlow that the model will be built into the default Graph.
	  with tf.Graph().as_default():
		# Generate placeholders for the images and labels.
		input_placeholder, output_placeholder = input_body.placeholder_inputs()

		# Build a Graph that computes predictions from the network model.
		input_size = self.input_body.train_in.shape[1]
		
		output_predictions = self.body_nework.dense_network(input_placeholder,
															input_size, 
		                                                    self.hidden1_units, 
		                                                    self.hidden2_units)

		# Add to the Graph the Ops for loss calculation.
		loss = self.body_nework.loss(output_placeholder, output_predictions)

		### Stopped here ### Stopped here### Stopped here### Stopped here###
		
		
		
		
		
		
		# Add to the Graph the Ops that calculate and apply gradients.
		train_op = mnist.training(loss, FLAGS.learning_rate)

		# Add the Op to compare the logits to the labels during evaluation.
		eval_correct = mnist.evaluation(logits, labels_placeholder)

		# Build the summary Tensor based on the TF collection of Summaries.
		summary = tf.summary.merge_all()

		# Add the variable initializer Op.
		init = tf.global_variables_initializer()

		# Create a saver for writing training checkpoints.
		saver = tf.train.Saver()

		# Create a session for running Ops on the Graph.
		sess = tf.Session()

		# Instantiate a SummaryWriter to output summaries and the Graph.
		summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

		# And then after everything is built:

		# Run the Op to initialize the variables.
		sess.run(init)

		# Start the training loop.
		for step in xrange(FLAGS.max_steps):
		  start_time = time.time()

		  # Fill a feed dictionary with the actual set of images and labels
		  # for this particular training step.
		  feed_dict = fill_feed_dict(data_sets.train,
		                             images_placeholder,
		                             labels_placeholder)

		  # Run one step of the model.  The return values are the activations
		  # from the `train_op` (which is discarded) and the `loss` Op.  To
		  # inspect the values of your Ops or variables, you may include them
		  # in the list passed to sess.run() and the value tensors will be
		  # returned in the tuple from the call.
		  _, loss_value = sess.run([train_op, loss],
		                           feed_dict=feed_dict)

		  duration = time.time() - start_time

		  # Write the summaries and print an overview fairly often.
		  if step % 100 == 0:
		    # Print status to stdout.
		    print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
		    # Update the events file.
		    summary_str = sess.run(summary, feed_dict=feed_dict)
		    summary_writer.add_summary(summary_str, step)
		    summary_writer.flush()

		  # Save a checkpoint and evaluate the model periodically.
		  if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
		    checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
		    saver.save(sess, checkpoint_file, global_step=step)
		    # Evaluate against the training set.
		    print('Training Data Eval:')
		    do_eval(sess,
		            eval_correct,
		            images_placeholder,
		            labels_placeholder,
		            data_sets.train)
		    # Evaluate against the validation set.
		    print('Validation Data Eval:')
		    do_eval(sess,
		            eval_correct,
		            images_placeholder,
		            labels_placeholder,
		            data_sets.validation)
		    # Evaluate against the test set.
		    print('Test Data Eval:')
		    do_eval(sess,
		            eval_correct,
		            images_placeholder,
		            labels_placeholder,
		            data_sets.test)
		
