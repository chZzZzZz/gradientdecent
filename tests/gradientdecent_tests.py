from nose.tools import *
from gradientdecent import LRGrad
import numpy as np



def test_LRGrad():
    X = [[1,2,3],
         [4,5,6],
         [3,4,5],
         [6,5,7],
         [8,7,9]]
    y=[1,2,2,3,4]
    lr = LRGrad(X, y, 'SGD', alpha=0.1, epoch=1)
    print(lr.X)
    assert_equal(lr.X.shape, (5,4))


