# main.py
import math

import numpy as np
import pandas as pd
from sklearn import datasets

from cs231nExercise.exercise1.KNNClassifier import NearestNeighbour

# 데이터 로드
digits = datasets.load_digits()
data = pd.DataFrame(np.c_[digits['data'], digits['target']])

# training / test split
# Randomly shuffle the index of nba.
random_indices = np.random.permutation(data.index)

# Set a cutoff for how many items we want in the test set (in this case 1/3 of the items)
test_cutoff = math.floor(len(data) / 3)

# Generate the test set by taking the first 1/3 of the randomly shuffled indices.
ts = data.loc[random_indices[:test_cutoff]]  # 주석에서는 1을 시작으로 했지만, 일반적으로 첫 번째부터 자르는 게 맞음

# Generate the train set with the rest of the data.
tr = data.loc[random_indices[test_cutoff:]]

# Object
nn = NearestNeighbour()

# training (just save)
nn.train(tr.iloc[:, :-1].values, tr.iloc[:, -1].values)

# 예측 진행 (테스트 셋의 첫 번째 열)
Ypred = nn.predict(ts.iloc[:, :-1].values)

# 예측 결과와 실제 레이블 비교 출력
print("Predicted labels:", Ypred)
print("Actual labels:", ts.iloc[:, -1].values)
