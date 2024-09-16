# K-NN classfication
import numpy as np
import matplotlib.pyplot as plt
import math

class NearestNeighbour:
    def __init__(self):
        pass

    def train(self, X, y):
        """X : N x training data set, Y : N x 1 label"""
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        """X : M x D test data set"""
        num_text = X.shape[0] # 행 개수 (M)
        Ypred = np.zeros(num_text, dtype=self.ytr.dtype) # 먼저 0으로 가득 채워둠

        for i in np.arange(len(X)): # 모든 test data에 대해
            print(i, '\n')

            distance = np.zeros(len(self.Xtr))
            for j in np.arange(len(self.Xtr)):
                distances = np.sum(np.abs(self.Xtr[j]- X[i, :]), axis=0) # training data와 L1-norm을 구함
                distance[j] = distances

            min_index = np.argmin(distance) # 가장 가까운 1-NN의 index를 구함
            Ypred[i] = self.ytr[min_index] # 가장 가까운 이웃의 label을 적용

        return(Ypred)

