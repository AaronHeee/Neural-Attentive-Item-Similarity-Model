from __future__ import absolute_import
from __future__ import division

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import logging
from time import time

from Dataset import Dataset
from Dataset import Data
from Evaluate import Evaluate

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run NAIS_prod.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='pinterest-20',
                        help='Choose a dataset.')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, 1: Pretrain with updating FISM variables, 2:Pretrain with fixed FISM variables.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size.')
    parser.add_argument('--embed_size', type=int, default=16,
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[1e-7,1e-7,1e-5]',
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--alpha', type=float, default=0,
                        help='Index of coefficient of embedding vector')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    return parser.parse_args()

class NAIS_prod:

    def __init__(self, num_items, dataset_name, pretrain, batch_size, learning_rate, embedding_size, alpha, regs):
        self.num_items = num_items
        self.pretrain = pretrain
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.learning_rate = learning_rate
        self.embedding_size = embedding_size
        self.weight_size = int(embedding_size / 2)
        self.alpha = alpha
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.eta_bilinear = regs[2]
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, None])	#the index of users
            self.num_idx = tf.placeholder(tf.float32, shape=[None, 1])	#the number of items rated by users
            self.item_input = tf.placeholder(tf.int32, shape=[None, 1])	  #the index of items
            self.labels = tf.placeholder(tf.float32, shape=[None,1])	#the ground truth

    def _create_variables(self):
        with tf.name_scope("embedding"):  # The embedding initialization is unknown now
            trainable_flag = (self.pretrain!=2)
            self.c1 = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01), name='c1', dtype=tf.float32, trainable=trainable_flag)
            self.c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2')
            self.embedding_Q_ = tf.concat([self.c1, self.c2], 0, name='embedding_Q_')
            self.embedding_Q = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),name='embedding_Q', dtype=tf.float32,trainable=trainable_flag)
            self.bias = tf.Variable(tf.zeros(self.num_items),name='bias',trainable=trainable_flag)

            # Variables for attention
            self.W = tf.Variable(tf.truncated_normal(shape=[self.embedding_size, self.weight_size], mean=0.0, stddev=tf.sqrt(tf.div(2.0, self.weight_size + self.embedding_size))),
                                                                name='Weights_for_MLP', dtype=tf.float32)
            self.b = tf.Variable(tf.truncated_normal(shape=[1, self.weight_size], mean=0.0, stddev=0.01),
                                                                name='Bias_for_MLP', dtype=tf.float32)
            self.h = tf.Variable(tf.truncated_normal(shape=[self.weight_size, 1], mean=0.0, stddev=0.01),
                                                                name='H_for_MLP', dtype=tf.float32)

    def _attention_MLP(self, q_):  #b*n*2e
        with tf.name_scope("attention_MLP"):
            self.W = tf.expand_dims(self.W, 0)
            b = tf.shape(q_)[0]
            n = tf.shape(q_)[1]
            self.W = tf.tile(self.W, tf.stack([b,1,1]))  #b*2e*w

            self.b = tf.expand_dims(self.b, 0)  #1*1*w
            MLP_output = tf.matmul(q_, self.W) + self.b #(b*n*2e) * (b*2e*w) + (1*1*w) = b*n*w
            self.h = tf.tile(tf.expand_dims(self.h, 0), tf.stack([b,1,1])) #b*w*1

            A_ = tf.reduce_sum(tf.matmul(MLP_output, self.h),2) #b*n

            # softmax for not mask features
            exp_A_ = tf.exp(A_)
            num_idx = tf.reduce_sum(self.num_idx, 1)
            mask_mat = tf.sequence_mask(num_idx, maxlen = n, dtype = tf.float32) #b*n
            exp_A_ = mask_mat * exp_A_
            exp_sum = tf.reduce_sum(exp_A_, 1, keep_dims=True)  #b*1
            #frac = tf.pow(exp_sum, -1) #b*1

            A = tf.expand_dims(tf.div(exp_A_, exp_sum),2) #b*n*1
            #A = tf.expand_dims(frac * exp_A_, 2, name="attention") #b*n*1
            return tf.reduce_sum(A * self.embedding_q_, 1)

    def _create_inference(self):
        with tf.name_scope("inference"):
            self.embedding_q_ = tf.nn.embedding_lookup(self.embedding_Q_, self.user_input) #b*n*e
            self.embedding_q = tf.nn.embedding_lookup(self.embedding_Q, self.item_input) #b*1*e
            self.embedding_p = self._attention_MLP(self.embedding_q_ * self.embedding_q)

            # self.embedding_p = self._attention_MLP(tf.nn.embedding_lookup(self.embedding_Q_, self.user_input))
            # self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q_, self.user_input), 1)
            self.embedding_q = tf.reduce_sum(self.embedding_q, 1)
            self.bias_i = tf.nn.embedding_lookup(self.bias, self.item_input)
            # self.coeff = tf.pow(self.num_idx, -tf.constant(self.alpha, tf.float32, [1]))
            self.output = tf.sigmoid(tf.expand_dims(tf.reduce_sum(self.embedding_p*self.embedding_q, 1),1) + self.bias_i)

    def _create_loss(self):
        with tf.name_scope("loss"):
            self.loss = tf.losses.log_loss(self.labels, self.output) + \
                        self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_Q)) + \
                        self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q_)) + \
                        self.eta_bilinear * tf.reduce_sum(tf.square(self.W))

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

def training(flag, model, dataset, batch_size, epochs, num_negatives):
    logging.info("begin training the NAIS_prod model...")
    
    saver = tf.train.Saver({'c1':model.c1,'embedding_Q':model.embedding_Q, 'bias':model.bias})

    weight_path = 'Pretrain/%s' % model.dataset_name
    with tf.Session() as sess:
        if flag != 0:
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(weight_path+'/checkpoint'))
            print weight_path + '/checkpoint'
            print ckpt.model_checkpoint_path
            if ckpt and ckpt.model_checkpoint_path:
                sess.run(tf.global_variables_initializer())
                saver.restore(sess, ckpt.model_checkpoint_path)
                logging.info("using pretrained variables")
                print "using pretrained variables"
        else:
            sess.run(tf.global_variables_initializer())
            logging.info("initialized")
            print "initialized"

        data = Data(dataset.trainMatrix, dataset.trainList, batch_size, num_negatives)
        evaluate = Evaluate(model, sess, dataset.trainList, dataset.testRatings, dataset.testNegatives)

        index = 0
        epoch_count = 0

        batch_begin = time()
        data.data_shuffle()
        batch_time = time() - batch_begin

        train_begin = time()

        while epoch_count < epochs:

            if data.last_batch:
                train_time = time() - train_begin

                loss_begin = time()
                train_loss = training_loss(model, sess, data)
                loss_time = time() - loss_begin

                eval_begin = time()
                (hits, ndcgs, losses) = evaluate.eval()
                hr, ndcg, test_loss = np.array(hits).mean(), np.array(ndcgs).mean(), np.array(losses).mean()
                eval_time = time() - eval_begin

                logging.info("Epoch %d [%.1fs + %.1fs]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1fs] train_loss = %.4f [%.1fs]" % (
                    epoch_count, batch_time, train_time, hr, ndcg, test_loss, eval_time, train_loss, loss_time))
                print "Epoch %d [%.1fs + %.1fs]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1fs] train_loss = %.4f [%.1fs]" % (
                    epoch_count, batch_time, train_time, hr, ndcg, test_loss, eval_time, train_loss, loss_time)
                batch_begin = time()
                data.data_shuffle()
                epoch_count += 1
                batch_time = time() - batch_begin

                train_begin = time()

            training_batch(index, model, sess, data)

            index += 1

def training_batch(index, model, sess, data):

    user_input, num_idx, item_input, labels = data.batch_gen(index)

    feed_dict = {model.user_input: user_input, model.num_idx: num_idx[:, None], model.item_input: item_input[:, None],
                 model.labels: labels[:, None]}
    batch_loss, _ = sess.run([model.loss, model.optimizer], feed_dict)


def training_loss(model, sess, data):
    index = 0
    train_loss = 0.0
    while index == 0 or data.last_batch == 0:
        user_input, num_idx, item_input, labels = data.batch_gen(index)
        feed_dict = {model.user_input: user_input, model.num_idx: num_idx[:, None], model.item_input: item_input[:, None],model.labels: labels[:, None]}
        train_loss += sess.run(model.loss, feed_dict)
        index += 1
    return train_loss / index

#(self, num_items, batch_size, learning_rate, embedding_size, lambda_bilinear, gamma_bilinear)
if __name__=='__main__':

    args = parse_args()
    regs = eval(args.regs)
    logging.basicConfig(filename="Log/NAIS_prod/log_lr%.4f_bs%d_%s_%.2f_pre%d" %(args.lr, args.batch_size, args.dataset, args.alpha, args.pretrain), level = logging.INFO)
    dataset = Dataset(args.path + args.dataset)
    model = NAIS_prod(dataset.num_items,args.dataset,args.pretrain,args.batch_size, args.lr, args.embed_size, args.alpha, regs)
    model.build_graph()
    training(args.pretrain, model, dataset, args.batch_size, args.epochs, args.num_neg)
