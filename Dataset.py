'''
Created on Aug 8, 2016
Processing datasets. 

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import scipy.sparse as sp
import numpy as np

class Dataset(object):
    '''
    Loading the data file
        trainMatrix: load rating records as sparse matrix for class Data
        trianList: load rating records as list to speed up user's feature retrieval
        testRatings: load leave-one-out rating test for class Evaluate
        testNegatives: sample the items not rated by user
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix = self.load_training_file_as_matrix(path + ".train.rating")
        self.trainList = self.load_training_file_as_list(path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        self.testNegatives = self.load_negative_file(path + ".test.negative")
        assert len(self.testRatings) == len(self.testNegatives)
        self.num_users, self.num_items = self.trainMatrix.shape

    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_training_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()
        print "already load the trainMatrix..."
        return mat

    def load_training_file_as_list(self, filename):
        # Get number of users and items
        u_ = 0
        lists, items = [], []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                if u_ < u:
                    lists.append(items)
                    items = []
                    u_ += 1
                items.append(i)
                line = f.readline()
        lists.append(items)
        print "already load the trainList..."
        return lists

class Data:
    '''
    Split the dataset into batches
    Represent user via the items' idx rated by user
    The num of items of batch matrix is determined by the longest user vector
    '''

    def __init__(self, trainMatrix, trainList, batch_size, num_negatives):
        self.train = trainMatrix
        self.list = trainList
        self.num_negatives = num_negatives
        self.batch_size = batch_size
        self.num_items, self.user_input, self.item_input, self.labels = self.get_train_data()
        self.iterations = len(self.user_input)
        self.index = np.arange(self.iterations)
        self.num_batch = self.iterations / self.batch_size
        self.epoch = 0

    def batch(self, i, IsOptimize):
        i = int(i % self.num_batch)
        # shuffle the dataset every epoch
        self.flag = ( i+1 == self.num_batch ) #judge the train loss batch is over or not
        if IsOptimize:
            if self.flag:
                self.epoch += 1
                self.num_items, self.user_input, self.item_input, self.labels = self.get_train_data()
                self.index_ = self.index
                np.random.shuffle(self.index)
            return self.get_train_batch(self.index, i)
        else:
            return self.get_train_batch(self.index_,i)

    def get_train_data(self):
        user_input, item_input, labels = [],[],[]
        num_items = self.train.shape[1]
        for (u, i) in self.train.keys():
            # positive instance
            user_items = []
            user_input.append(u)
            item_input.append(i)
            labels.append(1)
            # negative instances
            for t in xrange(self.num_negatives):
                j = np.random.randint(num_items)
                while self.train.has_key((u, j)):
                    j = np.random.randint(num_items)
                user_input.append(u)
                item_input.append(j)
                labels.append(0)
        return num_items, user_input, item_input, labels


    def get_train_batch(self, index,i):
        #represent the feature of users via items rated by him/her
        user_list, num_list, item_list, labels_list = [],[],[],[]
        begin = i * self.batch_size
        for idx in range(begin, begin+self.batch_size):
            user_idx = self.user_input[index[idx]]
            item_idx = self.item_input[index[idx]]
            nonzero_row = self.list[user_idx]
            nonzero_row = self.remove_item(nonzero_row, user_idx)
            user_list.append(nonzero_row)
            num_list.append(len(nonzero_row))
            item_list.append(item_idx)
            labels_list.append(self.labels[index[idx]])
        user_input = np.array(self.add_mask(self.num_items+1, user_list, num_list))
        num_idx = np.array(num_list)
        item_input = np.array(item_list)
        labels = np.array(labels_list)
        return user_input, num_idx, item_input, labels

    def remove_item(self, users, item):
        for i in range(len(users)):
            if users[i] == item:
                del users[i]
                break
        return users

    def add_mask(self, feature_mask, features, num):
        #uniformalize the length of each batch
        num_max = max(num)
        for i in xrange(len(features)):
            features[i] = features[i] + [feature_mask] * (num_max - len(features[i]))
        return features
