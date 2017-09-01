from __future__ import absolute_import
from __future__ import division

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cProfile
import tensorflow as tf
import numpy as np
import logging
from time import time

from Dataset import Dataset
from Dataset import Data
from Evaluate import Evaluate

import argparse

SKIP_STEP = 200

def parse_args():
    parser = argparse.ArgumentParser(description="Run FISM.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='pinterest-20',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--embed_size', type=int, default=8,
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[1e-7,1e-7]',
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--alpha', type=float, default=0,
                        help='Index of coefficient of embedding vector')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    return parser.parse_args()

class FISM:

    def __init__(self, num_items, batch_size, learning_rate, embedding_size, alpha, regs):
        self.num_items = num_items
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, None])	#the index of users
            self.num_idx = tf.placeholder(tf.float32, shape=[None, 1])	#the number of items rated by users
            self.item_input = tf.placeholder(tf.int32, shape=[None, 1])	  #the index of items
            self.labels = tf.placeholder(tf.float32, shape=[None,1])	#the ground truth

    def _create_variables(self):
        with tf.name_scope("embedding"):  # The embedding initialization is unknown now
            c1 = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01), #why [0, 3707)?
                                                 name='items_embeddings_for_users', dtype=tf.float32)
            c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size])
            self.embedding_Q_ = tf.concat([c1, c2], 0)
            self.embedding_Q = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                                                                name='items_embeddings_for_items', dtype=tf.float32)
            self.bias = tf.Variable(tf.zeros(self.num_items))

    def _create_inference(self):
        with tf.name_scope("inference"):
            self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q_, self.user_input), 1)
            self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, self.item_input), 1)
            self.bias_i = tf.nn.embedding_lookup(self.bias, self.item_input)
            self.coeff = tf.pow(self.num_idx, -tf.constant(self.alpha, tf.float32, [1]))
            self.output = tf.sigmoid(self.coeff * tf.expand_dims(tf.reduce_sum(self.embedding_p*self.embedding_q, 1),1) + self.bias_i)

    def _create_loss(self):
        with tf.name_scope("loss"):
            self.loss = tf.losses.log_loss(self.labels, self.output) + \
                        self.lambda_bilinear*tf.reduce_sum(tf.square(self.embedding_Q)) + self.gamma_bilinear*tf.reduce_sum(tf.square(self.embedding_Q_))
            # self.loss = tf.nn.l2_loss(self.labels - self.output) + \
            #             self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_Q)) + self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q_))

    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._create_optimizer()
        logging.info("already build the computing graph...")

def training(model, dataset, batch_size, epochs, num_negatives):
    logging.info("begin training the FISM model...")
    saver = tf.train.Saver()
    #mkdir('checkpoints') #undifined
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter('./graphs', sess.graph)
        writer.close()

        data = Data(dataset.trainMatrix, dataset.trainList, batch_size, num_negatives)
        # model, sess, trainList, testRatings, testNegatives,
        evaluate = Evaluate(model, sess, dataset.trainList, dataset.testRatings, dataset.testNegatives)

        t = time()
        index = 0
        total_loss = 0.0
        epoch_count = 0

        while epoch_count < epochs:

            if data.last_batch:
                training_loss(epoch_count, model, sess, data)
                data.data_shuffle()
                epoch_count += 1
                (hits, ndcgs, losses) = evaluate.eval()
                hr, ndcg, test_loss = np.array(hits).mean(), np.array(ndcgs).mean(), np.array(losses).mean()
                logging.info("epoch %d: hr = %.2f, ndcg = %.2f, test_loss = %.2f" % (epoch_count, hr, ndcg, test_loss))

            saver.save(sess, 'checkpoints/FISM', index)
            index += 1


# @profile
def training_batch(t, epoch, index, model, sess, data):
    # user_input, num_idx, item_input, labels = data.batch(index, IsOptimize=True)

    user_input, num_idx, item_input, labels = data.batch_gen(index)

    feed_dict = {model.user_input: user_input, model.num_idx: num_idx[:, None], model.item_input: item_input[:, None],
                 model.labels: labels[:, None]}
    batch_loss, _ = sess.run([model.loss, model.optimizer], feed_dict)
    if index%100 == 0:
        logging.info('(%.4f s) %d epoch: Batch loss at step %2d: %5.2f' % (
            time() - t, epoch, index, batch_loss))

def training_loss(epoch_count, model, sess, data):
    t = time()
    index = 0
    train_loss = 0.0
    while index == 0 or data.last_batch == 0:
        user_input, num_idx, item_input, labels = data.batch_gen(index)
        feed_dict = {model.user_input: user_input, model.num_idx: num_idx[:, None], model.item_input: item_input[:, None],model.labels: labels[:, None]}
        train_loss += sess.run(model.loss, feed_dict)
        index += 1
    logging.info('(%.4f s) epoch %d : train loss: %5.2f' % (time() - t, epoch_count, train_loss / index))

#(self, num_items, batch_size, learning_rate, embedding_size, lambda_bilinear, gamma_bilinear)
if __name__=='__main__':

    args = parse_args()
    regs = eval(args.regs)
    logging.basicConfig(filename="log_lr%.4f_bs%d" %(args.lr, args.batch_size), level = logging.INFO)
    dataset = Dataset(args.path + args.dataset)
    model = FISM(dataset.num_items, args.batch_size, args.lr, args.embed_size, args.alpha, regs)
    model.build_graph()
    training(model, dataset, args.batch_size, args.epochs, args.num_neg)
