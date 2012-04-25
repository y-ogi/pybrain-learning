# -*- coding: utf-8 -*-
import random

def f(x, y):
    return x and y

# ネットワーク作成
from pybrain.tools.shortcuts import buildNetwork
net = buildNetwork(2, 2, 1)

# テストデータ
from pybrain.datasets import SupervisedDataSet
ds = SupervisedDataSet(2, 1)
for i in range(1000):
    x = random.choice([0, 1])
    y = random.choice([0, 1])
    ds.addSample((x, y,), f(x, y),)

# 学習
from pybrain.supervised.trainers import BackpropTrainer
trainer = BackpropTrainer(net, ds, learningrate=0.01, momentum=0.99)
trainer.trainEpochs(3)
#trainer.trainUntilConvergence(validationProportion=0.10)

data = [(0, 0,),(0, 1,), (1, 0,), (1, 1,)]
for x, y in data:
    print '%d,%d=%f' % (x, y,  net.activate([x, y]),)

