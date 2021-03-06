# gradient decent implementation
[![](https://travis-ci.org/chZzZzZz/gradientdecent.svg?branch=master)](https://travis-ci.org/chZzZzZz/gradientdecent)

> An implementation of gradient decent solving **Regression** problems including Batch gradient decent,SGD and Mini-batch gradient decent

The goal of this project is to implementation the three gradient decent from scrach which makes me get more familiar with the gradient decent optimization algorithms. 
## Python Support
This project requires Python version 3.x(Python 2.x is not supported). 

## Installation

安装方法：进入项目目录在命令行输入

```
python setup.py install
```


## Usage example 使用示例
Using the Boston dataset from sklearn as an example

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gradientdecent import LRGrad
from sklearn.datasets import load_boston

boston = load_boston()
X,y = boston.data,boston.target

lr = LRGrad(X,y,'SGD',alpha=0.1,epoch=1)
history,best_theta = lr.run()
lr.plot_loss(history)
```
![image]( https://github.com/chZzZzZz/gradientdecent/raw/master/images/SGD.png )
## Documention
The current gradient decent contains the following params:

* param X : feature vectors of input
* param y : label of input
* param type : has three type "FULL"-Batch gradient decent "SGD"-SGD "MINI"-MINI-batch gradient decent
* param batch_size : the batch size of MINI-batch gradient decent
* param epoch : when using the SGD or MINI,every epoch iters update the loss
* param alpha : learning rate
* param shuffle : when |loss_new-loss|<shuffle,the iteration stop
* param theta : the param we want to learn
* param history : restore the iter and loss
* plot_loss () : plot the loss changes with iterations
* run () : run the gradient decent and finally return the history and best params we want to learn



## Authors 关于作者

* **chZzZzZz** - *Initial work* 
 
## References
[https://www.chzzz.club/post/144.html]

[http://sofasofa.io/tutorials/python_gradient_descent/]

## License 授权协议

这个项目 MIT 协议， 请点击 [LICENSE](LICENSE) 了解更多细节。