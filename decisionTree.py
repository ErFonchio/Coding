import numpy as np
import pandas as pd
from library import Map
import math
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.preprocessing import LabelEncoder  
from library import DecisionTree as DT
import time

'''Restaurant'''
dataset = pd.read_csv("restaurant.csv")
df = pd.DataFrame(dataset)

column_values = {}
for elem in dataset.columns:
    column_values[elem] = set()
    for elem1 in dataset[elem]:
        column_values[elem].add(elem1)

#colonna ouput, risposta positiva, risposta negativa
dt = DT('Wait', 'Yes', 'No')
attributes = dt.PopListValue(dataset.columns.tolist(), 'Wait')
dt.LearnDecisionTree(dataset, attributes, dataset, column_values)

'''Obesity'''
dataset = pd.read_csv("/Users/alessandrococcia/Downloads/ObesityDataSet.csv")

'''data manipulation'''
dataset = dataset.round(1)
dataset = dataset.sample(frac=1, ignore_index=True) #shuffle sample in the training set
dataset.replace(to_replace=("Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", "Overweight_Level_II"), value=0, inplace=True)
dataset.replace(to_replace=("Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"), value=1, inplace=True)


'''Mappo le stringhe in interi'''
m = Map()
dataset = pd.get_dummies(dataset, drop_first=True).astype(float)
dataset.sample(frac=1)

TRAIN_TEST_SPLIT_PERCENTAGE = 0.90
dataset_training = dataset[:int(len(dataset) * TRAIN_TEST_SPLIT_PERCENTAGE)]
dataset_test = dataset[int(len(dataset) * TRAIN_TEST_SPLIT_PERCENTAGE):]

d = {}
for elem in dataset_training.columns:
    d[elem] = set()
    for elem1 in dataset_training[elem]:
        d[elem].add(elem1)

#print(d)
start = time.time()
dt1 = DT('NObeyesdad', 1, 0)
attributes1 = dt1.PopListValue(dataset_training.columns.tolist(), 'NObeyesdad')
dt1.LearnDecisionTree(dataset_training, attributes1, dataset_training, d)
stop = time.time()

total_len = len(dataset)
training_len = len(dataset_training)
test_len = len(dataset_test)
count_me = 0
count_sklearn = 0

X = dataset_training.loc[:, dataset_training.columns != 'NObeyesdad']
Y = dataset_training.loc[:, dataset_training.columns == 'NObeyesdad']
Z = dataset_test.loc[:, dataset_test.columns != 'NObeyesdad']

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, Y)
ret = clf.predict(Z)

for i in range(training_len, total_len):
    input = dataset_test.loc[i:i]
    prediction = dt1.Prediction(input)
    if prediction == dataset_test['NObeyesdad'][i]:
        count_me += 1
    if ret[i-training_len] == dataset_test['NObeyesdad'][i]:
        count_sklearn += 1

#print(dataset)
print(count_me/test_len)
print(count_sklearn/test_len)

