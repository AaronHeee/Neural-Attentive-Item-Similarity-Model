import numpy as np

def swap(item):
    temp = item[0]
    item[0] = item[-1]
    item[-1] = temp

if __name__ =="__main__":

    a = np.array([[[1,2,2,1],[4,5,5,4],[7,8,8,7]],[[1,2,2,1],[4,5,5,4],[7,8,8,7]]])
    print a.shape
    b = np.zeros([2, 4, 3])
    print b.shape
    print np.mat(a,b)


    l = [1,2]
    print l
    swap(l)
    print l