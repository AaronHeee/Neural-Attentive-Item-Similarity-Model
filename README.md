# NAIS: Neural Attentive Item Similarity Model 

This is our official implementation for the paper:

**NAIS: Neural Attentive Item Similarity Model for Recommendation** 
Xiangnan He, Zhankui He, Jingkuan Song, Zhenguang Liu, Yu-Gang Jiang, & Tat-Seng Chua 
IEEE Transactions on Knowledge and Data Engineering (TKDE 2018)

Two collaborative filtering models: **NAIS_concat** and **NAIS_prod**. To target the models for implicit feedback and ranking task, we optimize them using log loss with negative sampling.

Also, we implement the baseline: **FISM**, which is the well-known item-based recommendation model.

**Please cite our paper if you use our codes. Thanks!**

 Corresponding Author: Dr. Xiangnan He (<http://www.comp.nus.edu.sg/~xiangnan/>)

## Quick to Start

Run NAIS_prod:

```shell
python NAIS.py --dataset pinterest-20 --pretrain 0 --weight_size 16 --embed_size 16 --data_alpha 0 --regs [0,0,1e-6] --alpha 0--beta 0.5 --lr 0.05 --algorithm 0
```

Run NAIS_concat:

```shell
python NAIS.py --dataset pinterest-20 --pretrain 0 --weight_size 16 --embed_size 16 --data_alpha 0 --regs [0,0,1e-6] --alpha 0--beta 0.5 --lr 0.05 --algorithm 1
```

Run FISM:

```shell
python FISM.py --dataset pinterest-20 --pretrain 0 --embed_size 16 --alpha 0 --lr 0.01
```

For more argument details, you can use `python FISM.py -h` and `python NAIS.py -h` to obtain them.

## Environment

Python 2.7

TensorFlow >= r1.0

Numpy >= 1.12

PS. For your reference, our server environment is Intel Xeon CPU E5-2630 @ 2.20 GHz and 64 GiB memory. We recommend your free memory is more than 16 GiB to reproduce our experiments (and we are still trying to reduce the memory cost...).

## Dataset

We provide two processed datasets: MovieLens 1 Million (ml-1m) and Pinterest (pinterest-20) in Data/

train.rating:

- Train file.
- Each Line is a training instance: userID\t itemID\t rating\t timestamp (if have)

test.rating:

- Test file (positive instances).
- Each Line is a testing instance: userID\t itemID\t rating\t timestamp (if have)

test.negative

- Test file (negative instances).
- Each line corresponds to the line of test.rating, containing 99 negative samples.
- Each line is in the format: (userID,itemID)\t negativeItemID1\t negativeItemID2 ...



Update: February 5, 2018
