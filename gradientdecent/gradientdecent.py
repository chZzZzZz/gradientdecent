import numpy as np
import matplotlib.pyplot as plt

class LRGrad:
    def __init__(self, X, y,type='FULL',batch_size=16, alpha=0.1, shuffle=0.1, theta = None,epoch=100):
        '''
        :param X:feature vectors of input
        :param y:label of input
        :param alpha:learning rate
        :param shuffle:when |loss_new-loss|<shuffle,the iteration stop
        :param theta:the param we want to learn
        '''
        self.X = np.array(X)
        self.y = np.array(y).reshape(-1,1)
        self.alpha = alpha
        self.shuffle = shuffle
        self.col = self.X.shape[1]+1#the number of columns of data
        self.m = self.X.shape[0]#the number of rows of data
        self.X = np.hstack((np.ones((self.m, 1)), self.X))#add a column of ones to X
        self.theta = np.ones((self.col,1))
        self.grad = np.ones((self.col,1))
        self.epoch = epoch
        self.type = type
        self.batch_size = batch_size

    #feature normalization
    def feature_normaliza(self):
        mu = np.zeros((1,self.X.shape[1]))
        sigma = np.zeros((1,self.X.shape[1]))
        mu = np.mean(self.X,axis=0)
        sigma = np.std(self.X,axis=0)
        for i in range(1, self.X.shape[1]):
            self.X[:,i] = (self.X[:,i]-mu[i])/sigma[i]
        return self.X
    #Compute grad
    def compute_grad(self):
        if self.type == 'FULL':
            h = np.dot(self.X,self.theta)
            self.grad = np.dot(np.transpose(self.X), h-self.y)/self.m
        elif self.type == 'SGD':
            r = np.random.randint(self.m)
            h = np.dot(np.array([self.X[r,:]]), self.theta)
            self.grad = np.dot(np.transpose(np.array([self.X[r,:]])), h - np.array([self.y[r,:]])) / self.m

        elif self.type == 'MINI':
            r = np.random.choice(self.m,self.batch_size,replace=False)
            h = np.dot(self.X[r,:], self.theta)
            self.grad = np.dot(np.transpose(self.X[r,:]), h - self.y[r,:]) / self.m
        else:
            print("NO such gradient dencent Method!")
        return self.grad

    def update_theta(self):
        self.theta = self.theta - self.alpha*self.grad
        return self.theta

    def compute_loss(self):
        hh = np.dot(self.X, self.theta)
        loss = np.sqrt(np.dot((np.transpose(hh-self.y)),(hh-self.y))/(2*self.m))
        return loss
    def run(self):
        self.X = self.feature_normaliza()
        self.grad = self.compute_grad()
        loss = self.compute_loss()
        self.theta = self.update_theta()
        loss_new = self.compute_loss()
        i = 1
        print('Round {} loss: {}'.format(i, np.abs(loss_new - loss)[0][0]))
        history = [[1], [loss_new[0][0]]]
        while np.abs(loss_new-loss)> self.shuffle:
            self.grad = self.compute_grad()
            self.theta = self.update_theta()
            if self.type == 'FULL':
                loss = loss_new
                loss_new = self.compute_loss()
            else:
                if i % self.epoch == 0:
                    loss = loss_new
                    loss_new = self.compute_loss()
            i += 1
            history[0].append(i)
            history[1].append(loss_new[0][0])
            print('Round {} loss: {}'.format(i, np.abs(loss_new-loss)[0][0]))
        best_theta = self.theta
        return history,best_theta
    def plot_loss(self,history):
        fig = plt.figure(figsize=(8, 6))
        plt.plot(history[0], history[1])
        plt.xlabel('Number of iterations')
        plt.ylabel('loss')
        plt.title([self.type + ' BATCH gradient decent' if self.type!='SGD' else 'SGD'][0])
        plt.show()



