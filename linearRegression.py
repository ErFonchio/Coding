import numpy as np
import pandas as pd
from library import Map
import math
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("/Users/alessandrococcia/Downloads/ObesityDataSet.csv")
'''sampling dataset'''
dataset = dataset.sample(frac=1)
'''mapping strings to numeric'''
map = Map()
dataset = map.mappingDataset(dataset)

ds = dataset.insert(0, "Bias", np.ones(len(dataset)), True) #Bias row

TRAIN_TEST_SPLIT_PERCENTAGE = 0.9
dataset_training = dataset[:int(len(dataset) * TRAIN_TEST_SPLIT_PERCENTAGE)]
dataset_test = dataset[int(len(dataset) * TRAIN_TEST_SPLIT_PERCENTAGE):]


ds = dataset_training
x = dataset_training.loc[:, dataset_training.columns != 'Weight']
y = dataset_training.loc[:, dataset_training.columns == 'Weight']

yStripped = np.float64(y)
xStripped = np.float64(x)

pseudoinverse = np.linalg.inv(np.matmul(xStripped.T, xStripped))

c = np.matmul(pseudoinverse, np.matmul(xStripped.T, yStripped))

'''TESTING'''
dt = dataset_test
dtX = dt.loc[:, dt.columns != 'Weight']
dtY = dt.loc[:, dt.columns == 'Weight']

dtY = np.float64(dtY)
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
scikit_predict = reg.predict(dtX)
print("MSE_sklearn: ", map.MSE(dtY, scikit_predict))
print("RMSE_sklearn: ", map.RMSE(dtY, scikit_predict))
print("MAE_sklearn: ", map.MAE(dtY, scikit_predict))
