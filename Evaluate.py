'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)

@author: hexiangnan
'''
import math
import heapq # for retrieval topK
import multiprocessing
import numpy as np
from time import time
#from numba import jit, autojit

# Global variables that are shared across processes

class Evaluate:

    def __init__(self, model, sess, trainList, testRatings, testNegatives, K = 10, num_thread = 1):
        self.model = model
        self.sess = sess
        self.trainList = trainList
        self.testRatings = testRatings
        self.testNegatives = testNegatives
        self.K = K
        self.num_thread = num_thread
        self.DictList = self.load_test_as_list()

    def load_test_as_list(self):
        DictList = []
        for idx in xrange(len(self.testRatings)):
            rating = self.testRatings[idx]
            items = self.testNegatives[idx]
            user = self.trainList[idx]
            num_idx_ = len(user)
            gtItem = rating[1]
            items.append(gtItem)
            # Get prediction scores
            num_idx = np.full(len(items),num_idx_, dtype=np.int32 )[:,None]
            user_input = []
            for i in range(len(items)):
                user_input.append(user)
            user_input = np.array(user_input)
            item_input = np.array(items)[:,None]
            feed_dict = {self.model.user_input: user_input, self.model.num_idx: num_idx, self.model.item_input: item_input}
            DictList.append(feed_dict)
        print("already load the evaluate model...")
        return DictList
    
    def eval(self):
        """
        Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
        Return: score of each test rating.
        """
        hits, ndcgs, losses = [],[],[]
        # if(self.num_thread > 1): # Multi-thread
        #     pool = multiprocessing.Pool(processes=self.num_thread)
        #     res = pool.map(self.eval_one_rating, range(len(self.testRatings)))
        #     pool.close()
        #     pool.join()
        #     hits = [r[0] for r in res]
        #     ndcgs = [r[1] for r in res]
        #     return (hits, ndcgs)
        # # Single thread
        num_hits = 0
        for idx in xrange(len(self.testRatings)):
            (hr,ndcg,loss) = self.eval_one_rating(idx)
            hits.append(hr)
            ndcgs.append(ndcg)
            losses.append(loss)
        return (hits, ndcgs, losses)
    
    #@profile
    def eval_one_rating(self, idx):

        map_item_score = {}
        rating = self.testRatings[idx]
        items = self.testNegatives[idx]
        gtItem = rating[1]
        labels = np.zeros(len(items))[:, None]
        labels[-1] = 1
        feed_dict = self.DictList[idx]
        feed_dict[self.model.labels] = labels
        predictions,loss = self.sess.run([self.model.output,self.model.loss], feed_dict = feed_dict)

        for i in xrange(len(items)):
            item = items[i]
            map_item_score[item] = predictions[i]
        # items.pop()
        # Evaluate top rank list

        ranklist = heapq.nlargest(self.K, map_item_score, key=map_item_score.get)
        hr = self.getHitRatio(ranklist, gtItem)
        ndcg = self.getNDCG(ranklist, gtItem)
        return (hr, ndcg, loss)

    def getHitRatio(self, ranklist, gtItem):
        for item in ranklist:
            if item == gtItem:
                return 1
        return 0

    def getNDCG(self, ranklist, gtItem):
        for i in xrange(len(ranklist)):
            item = ranklist[i]
            if item == gtItem:
                return math.log(2) / math.log(i+2)
        return 0
