import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gradientdecent import LRGrad
from sklearn.datasets import load_boston
#一元线性回归
# train = pd.read_csv('../data/train.csv')
# X=train[['id']]
# y=train['questions']

#多元线性回归
# data = pd.read_csv('../data/hpdata.csv',names=['area','num_rooms','price'])
# X = data[data.columns[:-1]]
# y = data[data.columns[-1]]

#sklearn数据集
boston = load_boston()

X,y = boston.data,boston.target
print(X.shape)
lr = LRGrad(X,y,'SGD',alpha=0.1,epoch=1)
history,best_theta = lr.run()
print(best_theta)
print(history)
lr.plot_loss(history)
