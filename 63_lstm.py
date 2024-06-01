
# coding: utf-8

# 

# Deep Learning
# =============
# 
# Assignment 6
# ------------
# 
# After training a skip-gram model in `5_word2vec.ipynb`, the goal of this notebook is to train a LSTM character model over [Text8](http://mattmahoney.net/dc/textdata) data.

# In[1]:

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve


# In[2]:

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % filename)
    else:
        print(statinfo.st_size)
        raise ValueError(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

filename = maybe_download('text8.zip', 31344016)


# In[3]:

def read_data(filename):
    f = zipfile.ZipFile(filename)
    for name in f.namelist():
        return tf.compat.as_str(f.read(name))
    f.close()
  
text = read_data(filename)
print('Data size %d' % len(text))


# Create a small validation set.

# In[4]:

valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size, valid_text[:64])


# Utility functions to map characters to vocabulary IDs and back.

# In[5]:

vocabulary_size = len(string.ascii_lowercase) + 1 # [a-z] + ' '
first_letter = ord(string.ascii_lowercase[0])

def char2id(char):
    if char in string.ascii_lowercase:
        return ord(char) - first_letter + 1
    elif char == ' ':
        return 0
    else:
        print('Unexpected character: %s' % char)
        return 0
  
def id2char(dictid):
    if dictid > 0:
        return chr(dictid + first_letter - 1)
    else:
        return ' '

print(char2id('a'), char2id('z'), char2id(' '), char2id('ï'))
print(id2char(1), id2char(26), id2char(0))


# Function to generate a training batch for the LSTM model.

# In[6]:

batch_size=64
num_unrollings=10

class BatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._text_size // batch_size
        self._cursor = [ offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch()
  
    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=float)
        for b in range(self._batch_size):
            batch[b, char2id(self._text[self._cursor[b]])] = 1.0
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch
  
    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for _ in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches

def characters(probabilities):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return [id2char(c) for c in np.argmax(probabilities, 1)]

def batches2string(batches):
    """Convert a sequence of batches back into their (most likely) string
    representation."""
    s = [''] * batches[0].shape[0]
    for b in batches:
        s = [''.join(x) for x in zip(s, characters(b))]
    return s

def batches2IDs(batches):
    """Convert a sequence of batches into the ids using char2id helper.
    """
    l = [ np.array( [ char2id(x) for x in characters(b) ] ) for b in batches ]
    return l

def builddictbigram():
    d = {}
    n = 0
    for i in range(vocabulary_size):
        for j in range(vocabulary_size):
            d[(i,j)] = n
            n+=1
    return d

dictbigram = builddictbigram()

def ids2bigrams(l0, l1):
    return np.array( [ dictbigram[x0, x1] for x0, x1 in zip(l0, l1) ] )

train_batches = BatchGenerator(train_text, batch_size, num_unrollings+1)
valid_batches = BatchGenerator(valid_text, 1, 1+1)

bx = train_batches.next()
bids = batches2IDs(bx)
print('train is list of unrollings+1, each a np.array(64,27) of a batch of one-hot encodings')
print('train batch:', len(bx), 'array of', bx[0].shape)
print('to-id is  list of unrollings+1, each a np.array(64) of a batch of id-embeddings')
print('to_id batch:', len(bids), 'array of', bids[0].shape)
print('to bigrams', ids2bigrams(bids[0],bids[1]))

v1 = valid_batches.next()
v2 = valid_batches.next()
print(batches2string(v1))
print(batches2string(v2))
print(ids2bigrams(batches2IDs(v1)[0],batches2IDs(v1)[1]))
print(ids2bigrams(batches2IDs(v2)[0],batches2IDs(v2)[1]))


# In[7]:

def logprob(predictions, labels):
    """Log-probability of the true labels in a predicted batch."""
    predictions[predictions < 1e-10] = 1e-10
    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution):
    """Sample one element from a distribution assumed to be an array of normalized
    probabilities.
    """
    r = random.uniform(0, 1)
    s = 0
    for i in range(len(distribution)):
        s += distribution[i]
        if s >= r:
            return i
    return len(distribution) - 1

def sample(prediction):
    """Turn a (column) prediction into 1-hot encoded samples."""
    p = np.zeros(shape=[1, vocabulary_size], dtype=float)
    p[0, sample_distribution(prediction[0])] = 1.0
    return p

def random_distribution():
    """Generate a random column of probabilities."""
    b = np.random.default_rng().uniform(0.0, 1.0, size=[1, vocabulary_size])
    return b/np.sum(b, 1)[:,None]


# ---
# Problem 1
# ---------
# 
# You might have noticed that the definition of the LSTM cell involves 4 matrix multiplications with the input, and 4 matrix multiplications with the output. Simplify the expression by using a single matrix multiply for each, and variables that are 4 times larger.
# 
# ---

# ** PROBLEM 1 DONE IN 6_lstm **
# 
# --------

# ---
# Problem 2
# ---------
# 
# We want to train a LSTM over bigrams, that is pairs of consecutive characters like 'ab' instead of single characters like 'a'. Since the number of possible bigrams is large, feeding them directly to the LSTM using 1-hot encodings will lead to a very sparse representation that is very wasteful computationally.
# 
# a- Introduce an embedding lookup on the inputs, and feed the embeddings to the LSTM cell instead of the inputs themselves.
# 
# b- Write a bigram-based LSTM, modeled on the character LSTM above.
# 
# c- Introduce Dropout. For best practices on how to use Dropout in LSTMs, refer to this [article](http://arxiv.org/abs/1409.2329).
# 
# ---
# 

# My Aproach:
# 
# Same as 6.2b:
# 
# 1) Identify the observation from the label.
# 
# 2) Unrolling are necesary for the LSTM cells.
# 
# 3) Split train from labels at feed.
# 
# 4) Generate num_unrollings + 2 batches.
# 
# 5) first is bigram0, second is bigram1, third is label.
# 
# 6) Train over bigram2ID(bigram0, bigram1)
# 
# 
# Plus the dropout.
# 
# 

# In[24]:

from tensorflow.python.framework import ops
#ops.reset_default_graph()

num_nodes = 64

graph = tf.Graph()
with graph.as_default():

    # Parameters:
    ifcox = tf.Variable(tf.truncated_normal([vocabulary_size**2, 4*num_nodes], -0.1, 0.1))
    ifcom = tf.Variable(tf.truncated_normal([num_nodes, 4*num_nodes], -0.1, 0.1))
    ifcob = tf.Variable(tf.zeros([1, 4*num_nodes]))
    # Variables saving state across unrollings.
    saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    # Classifier weights and biases.
    w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
    b = tf.Variable(tf.zeros([vocabulary_size]))
  
    keep_prob = tf.placeholder(tf.float32)

    # Definition of the cell computation.
    def lstm_cell(i, o, state):
        """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
        Note that in this formulation, we omit the various connections between the
        previous state and the gates."""
        
        # ORIGINAL CODE
        # was all-gates-state EQ tf.matmul(i, ifcox) + tf.matmul(o, ifcom) + ifcob
        # lookup instead of matmul. they are equivalent in this context.
        # was all-gates-state EQ tf.nn.embedding_lookup(ifcox, i) + tf.matmul(o, ifcom) + ifcob

        ifcoxdemb = tf.nn.embedding_lookup(ifcox, i)
        ifcoxd = tf.nn.dropout(ifcoxdemb, 1.0)
        od = tf.nn.dropout(o, keep_prob)
        all_gates_state = ifcoxd + tf.matmul(od, ifcom) + ifcob
        
        input_gate = tf.sigmoid(all_gates_state[:, 0:num_nodes])
        forget_gate = tf.sigmoid(all_gates_state[:, num_nodes: 2*num_nodes])
        update = all_gates_state[:, 2*num_nodes: 3*num_nodes]
        output_gate = tf.sigmoid(all_gates_state[:, 3*num_nodes:])

        state = forget_gate * state + input_gate * tf.tanh(update)
        return output_gate * tf.tanh(state), state

    # Input data. split input/labels
    train_inputs = list()
    train_labels = list()
    for _ in range(num_unrollings):
        train_inputs.append(tf.placeholder(tf.int64, shape=[batch_size])) # ids as input.
        train_labels.append(tf.placeholder(tf.int64, shape=[batch_size])) # ids as input.
    print('train_inputs: list of', len(train_inputs), train_inputs[0].get_shape())
    print('train_labels: list of', len(train_inputs), train_labels[0].get_shape())
    
    # Unrolled LSTM loop.
    outputs = list()
    output = saved_output
    state = saved_state
    for i in train_inputs:
        output, state = lstm_cell(i, output, state)
        outputs.append(output)

    # State saving across unrollings.
    with tf.control_dependencies([saved_output.assign(output), saved_state.assign(state)]):
        # Classifier.
        logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
        # use of sparse_softmax because it accepts int64s
        sofmacs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, tf.concat(0, train_labels))
        loss = tf.reduce_mean(sofmacs)


    # Optimizer.
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
       10.0, global_step, 2000, 0.5, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    momentum = tf.Variable(0.9)
    # other optimizers like tf.train.MomentumOptimizer(learning_rate, momentum)
    # other optimizer like tf.train.AdagradOptimizer(learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(
        zip(gradients, v), global_step=global_step)

    # Predictions.
    train_prediction = tf.nn.softmax(logits)
  
    # Sampling and validation eval: batch 1, no unrolling.
    
    # idem as train_data
    sample_input = tf.placeholder(tf.int64, shape = [1])    
    
    saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
    saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
    reset_sample_state = tf.group(
        saved_sample_output.assign(tf.zeros([1, num_nodes])),
        saved_sample_state.assign(tf.zeros([1, num_nodes])))
    sample_output, sample_state = lstm_cell(
        sample_input, saved_sample_output, saved_sample_state)
    with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                saved_sample_state.assign(sample_state)]):
        sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))


# In[25]:

num_steps = 20001
summary_frequency = 200
dropout_keep_prob = 0.5

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
with tf.Session(graph=graph, config=config) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    mean_loss = 0
    for step in range(num_steps):
        batches = train_batches.next()
        # transform input into IDs
        batchesIDs = batches2IDs( batches )
        feed_dict = dict()
        for i in range(num_unrollings):
            feed_dict[train_inputs[i]] = ids2bigrams(batchesIDs[i], batchesIDs[i+1])
            feed_dict[train_labels[i]] = batchesIDs[i+2]
        feed_dict[keep_prob] = dropout_keep_prob
        _, l, predictions, lr = session.run(
            [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
        mean_loss += l
        if step % summary_frequency == 0:
            if step > 0:
                mean_loss = mean_loss / summary_frequency
            # The mean loss is an estimate of the loss over the last few batches.
            print(
                'Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
            mean_loss = 0
            labels = np.concatenate(list(batches)[2:])
            print('Minibatch perplexity: %.2f' % float(
                np.exp(logprob(predictions, labels))))
            if step % (summary_frequency * 10) == 0:
                # Generate some samples.
                print('=' * 80)
                for _ in range(5):
                    feed0 = sample(random_distribution())
                    sentence = characters(feed0)[0]
                    feed1 = sample(random_distribution())
                    sentence += characters(feed1)[0]
                    reset_sample_state.run()
                    for _ in range(78):
                        # transform input into IDs
                        # batches2IDs accepts list. we need to wrap and unwrap.
                        bIDs = batches2IDs( [feed0, feed1] )
                        feedID = ids2bigrams(bIDs[0], bIDs[1])
                        prediction = sample_prediction.eval({sample_input: feedID, keep_prob: 1.0})
                        feed = sample(prediction)
                        sentence += characters(feed)[0]
                        feed0 = feed1
                        feed1 = feed
                    print(sentence)
                print('=' * 80)
            # Measure validation set perplexity.
            reset_sample_state.run()
            valid_logprob = 0
            for _ in range(valid_size):
                b = valid_batches.next()
                # transform input into IDs
                bIDs = batches2IDs(b)
                predictions = sample_prediction.eval({sample_input: ids2bigrams(bIDs[0],bIDs[1]), keep_prob: 1.0})
                valid_logprob = valid_logprob + logprob(predictions, b[2])
            print('Validation set perplexity: %.2f' % float(np.exp(
                valid_logprob / valid_size)))


# ---
# Problem 3
# ---------
# 
# (difficult!)
# 
# Write a sequence-to-sequence LSTM which mirrors all the words in a sentence. For example, if your input is:
# 
#     the quick brown fox
#     
# the model should attempt to output:
# 
#     eht kciuq nworb xof
#     
# Refer to the lecture on how to put together a sequence-to-sequence model, as well as [this article](http://arxiv.org/abs/1409.3215) for best practices.
# 
# ---
