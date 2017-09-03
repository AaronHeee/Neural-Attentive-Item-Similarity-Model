from Dataset import Dataset
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    #movieLen dataset
    ml = Dataset('./Data/ml-1m')
    ml_dict = {}
    for i in range(ml.num_users):
        num = len(ml.trainList[i])
        if num in ml_dict.keys():
            ml_dict[num] += 1
        else:
            ml_dict[num] = 1

    ml_dict_sorted = sorted(ml_dict.iteritems(), key=lambda d: d[1], reverse=True)

    print ml_dict_sorted

    f = open("ml_out.txt", "w")
    for item in ml_dict_sorted:
        print >> f, item
    f.close()

    # x = np.array(list(ml_dict.keys()))
    # y = np.array(list(ml_dict.values()))
    #
    # plt.plot(x,y)
    # plt.show()