import pandas as pd
import numpy as np
import math
from library import Map, GradientDescent
import time
import tensorflow as tf
from sklearn.linear_model import SGDClassifier


dataset = pd.read_csv("/Users/alessandrococcia/Downloads/ObesityDataSet.csv")
#dataset = dataset.sample(frac=1) #shuffle sample in the training set

'''Inserimento colonna di bias'''
dataset.insert(0, "Bias", np.ones(len(dataset)), True) #Bias row

'''mapping delle stringhe'''
m = Map()
dataset = m.mappingDataset(dataset)

TRAIN_TEST_SPLIT_PERCENTAGE = 0.9
dataset_training = dataset[:int(len(dataset) * TRAIN_TEST_SPLIT_PERCENTAGE)]
dataset_test = dataset[int(len(dataset) * TRAIN_TEST_SPLIT_PERCENTAGE):]

gd = GradientDescent(1e-5, 100, dataset_training.shape[1]-1, 'Weight')

start = time.time()
pesi = gd.SGD(dataset_training)
end = time.time()
#print(end-start, pesi)

'''sklearn'''
#X = dataset_training.loc[:, dataset_training.columns != 'Weight']
#Y = dataset_training.loc[:, dataset_training.columns == 'Weight']
#
#clf = SGDClassifier(max_iter = 1, tol=1e-3)
#clf.fit(X, Y)

