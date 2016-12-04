# Adam for mnist dataset.
from __future__ import absolute_import
from __future__ import division

import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Basic model parameters as external flags.
args = None

# We can't initialize these variables to 0 - the network will get stuck.
def weight_variable(shape):
  """Create a weight variable with appropriate initialization."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  """Create a bias variable with appropriate initialization."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.scalar_summary('stddev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  """Reusable code for making a simple neural net layer.

  It does a matrix multiply, bias add, and then uses relu to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read,
  and adds a number of summary ops.
  """
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights, layer_name + '/weights')
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases, layer_name + '/biases')
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.histogram_summary(layer_name + '/pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    tf.histogram_summary(layer_name + '/activations', activations)
    return activations

def inspect_grads(grad, var):
  if "layer1" in var.op.name:
    print var.op.name, grad.op.name
    return



def train():
  # Import data
  mnist = input_data.read_data_sets(args.data_dir,
                    one_hot=True)

  sess = tf.InteractiveSession()

  # Create a multilayer model.
  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.image_summary('input', image_shaped_input, 10)



  hidden1 = nn_layer(x, 784, 500, 'layer1')

  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.scalar_summary('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

  # Do not apply softmax activation yet, see below.
  y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

  with tf.name_scope('cross_entropy'):
    # The raw formulation of cross-entropy,
    #
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    #                               reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the
    # raw outputs of the nn_layer above, and then average across
    # the batch.
    diff = tf.nn.softmax_cross_entropy_with_logits(y, y_)
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)
    tf.scalar_summary('cross entropy', cross_entropy)

  with tf.name_scope('train'):
    # ----- [Rui] How we typically imexplicitly set up the optimizer
    # train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(
    #     cross_entropy)

    # ----- [Rui] How we cal the gradients and manually apply them
    optimizer_def = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    # [Rui] optimizer.compute_gradients() get you the grad of the optimzier(loss) w.r.t. each variable; then you can use optimizer.apply_gradients() to apply grads to the vars
    # [Rui] If you need the grad of a var w.r.t. another, you can look at the answer at http://stackoverflow.com/questions/35226428/how-do-i-get-the-gradient-of-the-loss-at-a-tensorflow-variable
    # and then you will be able to compute your own gradients of any variable (including the loss) w.r.t its previous variables;
    # and replacing the optimizer.apply_gradients() by adding an operation to each variable by tf.add().
    gvs = optimizer_def.compute_gradients(cross_entropy, colocate_gradients_with_ops=True)
    # [Rui] if you need to do something to the gradients (select part of them; clipping gradients, etc.)
    # def clip_recog_p(grad, var):
    #   if "recog_p" in var.op.name:
    #     return (tf.clip_by_value(grad, -0.001, 0.001), var)
    #   else:
    #     return (grad, var)
    # capped_gvs = [clip_recog_p(grad, var) for grad, var in gvs ]

    for grad, var in gvs:
      inspect_grads(grad, var)
    # ----- [Rui] apply the grdients of loss w.r.t. variables you need with the optimizer; you can get rid of this function by manually calaulating each gradients and applying them by tf.add(); 
    # then you will not be able to use built-in optimizers like SGD or Adam any more;
    train_step = optimizer_def.apply_gradients(gvs)


  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary('accuracy', accuracy)

  # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
  merged = tf.merge_all_summaries()
  tf.initialize_all_variables().run()

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries

  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train:
      xs, ys = mnist.train.next_batch(100)
      k = args.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  for i in range(args.max_steps):
    if i % 10 == 0:  # Record summaries and test-set accuracy
      _, acc, cross_entropy_output = sess.run([merged, accuracy, cross_entropy], feed_dict=feed_dict(False))
      print('%s\t%s\t%f' % (i, 1.0-acc, cross_entropy_output))
    # else:  # Record train set summaries, and train
    #   if i % 100 == 99:  # Record execution stats
    #     run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #     run_metadata = tf.RunMetadata()
    #     sess.run([merged, train_step],
    #               feed_dict=feed_dict(True),
    #               options=run_options,
    #               run_metadata=run_metadata)

    #     print('Adding run metadata for', i)
    else:  # Record a summary
      sess.run([merged, train_step], feed_dict=feed_dict(True))
    # else:  # Record train set summaries, and train
    #   if i % 100 == 99:  # Record execution stats
    #     run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #     run_metadata = tf.RunMetadata()
    #     sess.run([merged, train_step],
    #               feed_dict=feed_dict(True),
    #               options=run_options,
    #               run_metadata=run_metadata)

    #     print('Adding run metadata for', i)
    #   else:  # Record a summary
    #     sess.run([merged, train_step], feed_dict=feed_dict(True))




if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--max_steps', type=int, default=1000,
            help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
            help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
            help='Keep probability for training dropout.')
  parser.add_argument('--data_dir', type=str, default='tmp/data',
            help='Directory for storing data')
  args = parser.parse_args()

  train()
#EOF.
