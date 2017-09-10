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
_model = None
_testRatings = None
_testNegatives = None
_K = None
_DictList = None
_sess = None

def init_evaluate_model(model, sess, testRatings, testNegatives, trainList):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _trainList
    global _DictList
    global _sess
    _sess = sess
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _trainList = trainList
    return load_test_as_list()

def eval(model, sess, testRatings, testNegatives, DictList):

    global _model
    global _testRatings
    global _testNegatives
    global _K
    global _DictList
    global _sess
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _DictList = DictList
    _sess = sess
    _K = 10

    num_thread = 1#multiprocessing.cpu_count()
    hits, ndcgs, losses = [],[],[]
    if(num_thread > 1): # Multi-thread
        pool = multiprocessing.Pool(num_thread)
        res = pool.map(_eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        losses = [r[2] for r in res]
    # Single thread
    else:
        for idx in xrange(len(_testRatings)):
            (hr,ndcg, loss) = _eval_one_rating(idx)
            hits.append(hr)
            ndcgs.append(ndcg)  
            losses.append(loss)    
    return (hits, ndcgs, losses)

def load_test_as_list():
    DictList = []
    for idx in xrange(len(_testRatings)):
        rating = _testRatings[idx]
        items = _testNegatives[idx]
        user = _trainList[idx]
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
        feed_dict = {_model.user_input: user_input, _model.num_idx: num_idx, _model.item_input: item_input}
        DictList.append(feed_dict)
    print("already load the evaluate model...")
    return DictList

def _eval_one_rating(idx):

    map_item_score = {}
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    gtItem = rating[1]
    labels = np.zeros(len(items))[:, None]
    labels[-1] = 1
    feed_dict = _DictList[idx]
    feed_dict[_model.labels] = labels
    predictions,loss = _sess.run([_model.output,_model.loss], feed_dict = feed_dict)

    for i in xrange(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    # items.pop()
    # Evaluate top rank list

    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = _getHitRatio(ranklist, gtItem)
    ndcg = _getNDCG(ranklist, gtItem)
    return (hr, ndcg, loss)

def _getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def _getNDCG(ranklist, gtItem):
    for i in xrange(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
