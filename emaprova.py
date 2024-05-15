
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

dataset = pd.read_csv("/Users/alessandrococcia/Downloads/ObesityDataSet.csv")


# # About The Dataset
# The data consist of the estimation of obesity levels in people from the countries of Mexico, Peru and Colombia, with ages between 14 and 61 and diverse eating habits and physical condition , data was collected using a web platform with a survey where anonymous users answered each question, then the information was processed obtaining 17 attributes and 2111 records.
# The attributes related with eating habits are: Frequent consumption of high caloric food (**FAVC**), Frequency of consumption of vegetables (**FCVC**), Number of main meals (**NCP**), Consumption of food between meals (**CAEC**), Consumption of water daily (**CH20**), and Consumption of alcohol (**CALC**). The attributes related with the physical condition are: Calories consumption monitoring (**SCC**), Physical activity frequency (**FAF**), Time using technology devices (**TUE**), Transportation used (**MTRANS**)
# variables obtained :
# **Gender**, **Age**, **Height** and **Weight**.
# 
# **NObesity** values are:
# 
# * Underweight
# * Normal
# * Overweight
# * Obesity I
# * Obesity II
# * Obesity III

# In[40]:


#replace non-numerical values with numerical ones in the dataset

# One-hot encode the columns using pandas
dataset = pd.get_dummies(dataset, drop_first=True)

# manually replace values
#dataset_new = dataset.replace(to_replace=("Female", "Male"), value=(0, 1))
# Attention! replace is applied to all columns, manually check coeherence in subsitutions
#dataset_new.replace(to_replace=("yes", "no"), value=(1,0), inplace=True)
#dataset_new.replace(to_replace=("Sometimes", "Frequently", "Always"), value=(0.25, 0.5, 1), inplace=True)
#dataset_new.replace(to_replace=("Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"), value=(0,1,2,3,4), inplace=True)
#dataset_new.replace(to_replace=("Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"), value=(0,1,2,3,4,5,6), inplace=True)

#shuffle the dataframe, otherwhise they are sorted by obesity type
dataset = dataset.sample(frac=1, ignore_index=True)

dataset =(dataset-dataset.mean())/dataset.std()
'''Bias'''
dataset.insert(0, "Bias", np.ones(len(dataset)), True) #Bias row

# In[41]:


TRAIN_TEST_SPLIT_PERCENTAGE = 0.9
dataset_training = dataset[:int(len(dataset) * TRAIN_TEST_SPLIT_PERCENTAGE)]
dataset_test = dataset[int(len(dataset) * TRAIN_TEST_SPLIT_PERCENTAGE):]
#dataset_new


# In[122]:


# LINEAR REGRESSION w/ Gradient Descent TRAINING

# Use numpy indexing to extract sub-matrices X (inputs) and Y (outputs)
# Third column is the wanted result
#X = np.delete(dataset_training_numpy, 3, 1)
#Y = dataset_training_numpy[:, 3]

# extract X and Y matrices. For this model, we are interested in calculating the Weight
lr_X = dataset_training.loc[:, dataset_training.columns != 'Weight']
lr_Y = dataset_training.loc[:, dataset_training.columns == 'Weight']


## normalize X
#n_lr_X = np.array(( (lr_X - lr_X.mean()) / lr_X.std()))
n_lr_X = lr_X
n_lr_Y = lr_Y
#
## normalize Y
#lr_Y = np.array(lr_Y)
#n_lr_Y = (lr_Y - lr_Y.mean()) / lr_Y.std()
#
## add bias column
#n_lr_X = np.c_[np.ones(len(n_lr_X)), n_lr_X]

'''
    w <- any point in the parameters space
    while not converged do
        for each w_i in w do
            w_i <- w_i - a * d/dw_i Loss(w)

    in the case of multi-variable model

            wi <- wi + a \sum_j (y_j - h_w(x_j)) * x_{j,i}

    where
         - h_w is the guess at the current epoch
         - a is the learning rate
'''


rows = n_lr_X.shape[0]
columns = n_lr_X.shape[1]

print()

lrn_rate = 0.001
# start with w as a random vector in the parameter space
w = np.random.rand(columns, 1)
#w = np.ones((columns, 1))

#print(w.shape, "Initial w", w)
# TODO: better check for convergence
epocs = 100000
for times in range(0, epocs):
    #print(n_lr_X, w, n_lr_Y)
    #print(n_lr_X.shape, w.shape, n_lr_Y.shape)
    #print(np.array(n_lr_X@w)-np.array(n_lr_Y))

    gradient = ( (n_lr_X.T)@(np.array(n_lr_X@w)-np.array(n_lr_Y) ))

    #print(np.array(gradient))
    w -= lrn_rate * gradient / rows

#print(w)
            


# In[125]:


# LINEAR REGRESSION TESTING

# Use numpy indexing to extract sub-matrices X (inputs) and Y (outputs)
# The last column is the wanted result
lr_X_test = np.array(dataset_test.loc[:, dataset_test.columns != 'Weight'])
lr_Y_test = np.array(dataset_test.loc[:, dataset_test.columns == 'Weight'])


n_lr_X_test = lr_X_test
n_lr_Y_test = lr_Y_test
#n_lr_X_test = np.array(( (lr_X_test - lr_X_test.mean()) / lr_X_test.std()))
#n_lr_X_test = np.c_[np.ones(len(n_lr_X_test)), n_lr_X_test]
#n_lr_Y_test = np.array(( (lr_Y_test - lr_Y_test.mean()) / lr_Y_test.std()))

# Test using sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

print("\n######### LINEAR REGRESSION w/ Gradient Descent #########")
print("\n######### My model #########")
lr_Y_predicted = n_lr_X_test@w

# MSE
MSE = mean_squared_error(n_lr_Y_test, lr_Y_predicted)
print("MSE: " + str(MSE))
# RMSE
RMSE = sqrt(MSE)
print("RMSE: " + str(RMSE))
# MAE
MAE = mean_absolute_error(n_lr_Y_test, lr_Y_predicted)
print("MAE: " + str(MAE))

print("\n######### Scikit learn w/ SGD #########")
from sklearn.linear_model import SGDRegressor
# Definisci il regressore SGD
sgd_regressor = SGDRegressor(max_iter=100000, alpha=0.001, random_state=42)
# Addestra il modello
sgd_regressor.fit(n_lr_X, np.ravel(n_lr_Y))
# Effettua previsioni
lr_Y_predicted_sk = np.array(sgd_regressor.predict(n_lr_X_test))

# MSE
MSE_ = mean_squared_error(n_lr_Y_test, lr_Y_predicted_sk)
print("MSE: " + str(MSE_))
# RMSE
RMSE_ = sqrt(MSE_)
print("RMSE: " + str(RMSE_))
# MAE
MAE_ = mean_absolute_error(n_lr_Y_test, lr_Y_predicted_sk)
print("MAE: " + str(MAE_))
