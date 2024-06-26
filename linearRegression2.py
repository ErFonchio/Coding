import numpy as np
import pandas as pd
from library import Map
import math
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("/Users/alessandrococcia/Downloads/ObesityDataSet.csv")

'''sampling dataset'''
dataset = dataset.sample(frac=1, ignore_index=True)

'''testing errors'''
map = Map()

dataset = pd.get_dummies(dataset, drop_first=True, dtype=float)

'''normalization'''
dataset = (dataset-dataset.mean())/dataset.std()

dataset.insert(0, "Bias", np.ones(len(dataset)), True) #Bias row

TRAIN_TEST_SPLIT_PERCENTAGE = 0.9
dataset_training = dataset[:int(len(dataset) * TRAIN_TEST_SPLIT_PERCENTAGE)]
dataset_test = dataset[int(len(dataset) * TRAIN_TEST_SPLIT_PERCENTAGE):]

x = dataset_training.loc[:, dataset_training.columns != 'Weight']
y = dataset_training.loc[:, dataset_training.columns == 'Weight']

yStripped = y.values
xStripped = x.values

pseudoinverse = np.linalg.inv(np.matmul(xStripped.T, xStripped))
c = np.matmul(pseudoinverse, np.matmul(xStripped.T, yStripped))

'''TESTING'''
dt = dataset_test
dtX = dt.loc[:, dt.columns != 'Weight']
dtY = dt.loc[:, dt.columns == 'Weight']

dtY = dtY.values
predizione = np.matmul(dtX, c)

'''Stima errore'''
mse = map.MSE(dtY, predizione)
rmse = map.RMSE(dtY, predizione)
mae = map.MAE(dtY, predizione)

print("MSE: ", mse)
print("RMSE: ", rmse)
print("MAE: ", mae)

'''sklearn'''
reg = LinearRegression().fit(xStripped, yStripped)
scikit_predict = reg.predict(dtX.values)

print("MSE_sklearn: ", map.MSE(dtY, scikit_predict))
print("RMSE_sklearn: ", map.RMSE(dtY, scikit_predict))
print("MAE_sklearn: ", map.MAE(dtY, scikit_predict))
