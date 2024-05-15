import numpy as np
import math
import random as rn
import pandas as pd
from library import Map, GradientDescent
import time
from random import randrange

#profiling libraries
import cProfile
import pstats

'''Downloading dataset'''
dataset = pd.read_csv("/Users/alessandrococcia/Downloads/ObesityDataSet.csv")

'''Dataset sampling'''
dataset = dataset.sample(frac=1, ignore_index=True) #shuffle sample in the training set

'''mapping delle stringhe'''
dataset.replace(to_replace=("Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", "Overweight_Level_II"), value=0, inplace=True)
dataset.replace(to_replace=("Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"), value=1, inplace=True)
dataset = pd.get_dummies(dataset, drop_first=True, dtype=float)

'''normalization'''
dataset = (dataset-dataset.mean())/dataset.std()

'''Inserimento colonna di bias'''
dataset.insert(0, "Bias", np.ones(len(dataset)), True) #Bias row


TRAIN_TEST_SPLIT_PERCENTAGE = 0.9
dataset_training = dataset[:int(len(dataset) * TRAIN_TEST_SPLIT_PERCENTAGE)]
dataset_test = dataset[int(len(dataset) * TRAIN_TEST_SPLIT_PERCENTAGE):]

output_string = 'NObeyesdad'
input_totale = dataset_training.loc[:len(dataset_training), dataset_training.columns != output_string]
output_totale = dataset_training.loc[:len(dataset_training), dataset_training.columns == output_string]

'''Values to determine threshold'''
positive_value, negative_value = max(output_totale['NObeyesdad'].unique()), min(output_totale['NObeyesdad'].unique())
threshold = 0.5

class NeuralNetwork:
    def __init__(self, learning_rate, len_input, num_layers, num_layer_nodes, epochs=1):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.len_input = len_input
        self.num_layers = num_layers
        self.num_layer_nodes = num_layer_nodes
        self.weight_list = []
        self.layer_list = []
        self.derivate_list = []
        self.threshold = 0.5
        self.totalerror = 0


        #inizializzazione matrice dei pesi
        for i in range(self.num_layers-1):
            new_matrix = np.random.rand(self.num_layer_nodes[i], self.num_layer_nodes[i+1]) #matrice con righe=i e colonne i+1: (input,output) di un layer
            self.weight_list.append(new_matrix)

        #inizializzazione layer
        for i in range(self.num_layers):
            self.layer_list.append(np.zeros(self.num_layer_nodes[i]))

        #inizializzazione lista derivate parziali
        for i in range(self.num_layers-1): #non viene contato il layer di input
            self.derivate_list.append(np.zeros(self.num_layer_nodes[i+1]))

    def activation(self, x, type='sigmoid'): 
        if type=='sigmoid':
            return 1/(1+np.exp(-x))
        elif type=='tanh':
            return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        return None

    def activation_derivative(self, x, type='sigmoid'):
        if type=='sigmoid':
            return (1-self.activation(x, type='sigmoid'))*self.activation(x, type='sigmoid')
        return None
    
    def training(self, dataset_training):
        for e in range(self.epochs):
            print("Epoca numero", e)
            for i in range(dataset_training.shape[0]): #ciclo che itera per ogni sample del dataset 
                sample = dataset_training.loc[i:i]
                self.feedforwarding(sample)
                #print("Error: ", self.totalerror)
                self.backpropagation(i)


    def testing(self, dataset_testing) -> list: 
        predicted_out = []
        for i in range(dataset_testing.shape[0]):
            self.feedforwarding(dataset_testing.loc[i:i])
            predicted_out.append(n.layer_list[-1][0][0])

        return predicted_out

    def feedforwarding(self, sample):
        '''fase di feedforwarding'''
        self.layer_list[0] = np.array(sample) #inizializzo l'input della rete ai valori del sample
        for layer_index in range(self.num_layers-1): # per ogni weight_matrix esistente
            out_unactivated = np.dot(self.layer_list[layer_index], self.weight_list[layer_index])#+0.25 #+ bias test
            #print(out_unactivated)
            self.layer_list[layer_index+1] = self.activation(out_unactivated)
            #print(self.layer_list)

    def backpropagation(self, i): 
        '''fase di backtracking'''
        
        #inizializzazione errore output
        error = -(self.activation(output_totale.loc[i:i])-self.layer_list[-1])
        self.derivate_list[-1] = error
        for j in range(num_layers-2, 0, -1): #3-2 layer: si parte dall'indice 1 con 0 escluso (es.)
            
            ad = self.activation_derivative(self.layer_list[j])
            summatory = np.matmul(self.weight_list[j], self.derivate_list[j])
            self.derivate_list[j-1] = np.multiply(ad, summatory.T).T
            '''aggiornamento dei pesi'''
            self.weight_list[j] -= self.learning_rate*np.outer(self.layer_list[j], self.derivate_list[j])

        #manca l'aggiornamento del peso all'indice 0 che non è calcolato nell'ultimo ciclo
        self.weight_list[0] -= learning_rate*np.outer(self.layer_list[0], self.derivate_list[0])

    def binary_cross_entropy(y_true, y_pred):
        epsilon = 1e-15  # Piccola costante per evitare divisioni per zero
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip dei valori per evitare valori di logaritmo negativi
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        # Calcolo dell'errore per un singolo esempio di training o testing
        error = binary_cross_entropy(y_true, y_pred)

epochs = 10
learning_rate = 0.02
len_input = len(input_totale.columns)
num_layer_nodes = [len_input, 1000, 1] #numero nodi di ciascun layer da sinistra verso destra
num_layers = len(num_layer_nodes) #numero di layer compresi nella rete neurale input e output compresi

# creo un layer di 10 features -> 2 matrici di pesi: len_input x 10 e 10 x 1
# considerando l'ouput di un nodo per determinare la classificazione




n = NeuralNetwork(learning_rate=learning_rate, len_input=len_input, 
                num_layers=num_layers, num_layer_nodes=num_layer_nodes, epochs=epochs
                )
n.training(input_totale)


'''fase di testing'''

dataset_test = dataset_test.reset_index(drop=True)
X_test = dataset_test.loc[:, dataset_test.columns != 'NObeyesdad']
Y_effettivo = np.array(dataset_test['NObeyesdad'])


'''fase di testing'''
predicted_out = np.array(n.testing(X_test))
real_out = np.array(1/(1+np.exp(-Y_effettivo)))


upper_value = 1
lower_value = 0
threshold = 0.5

print(max(predicted_out), min(predicted_out))


# polarizzazione
predicted_out[predicted_out > threshold] = upper_value
predicted_out[predicted_out <= threshold] = lower_value
real_out[real_out > threshold] = upper_value
real_out[real_out <= threshold] = lower_value

count = 0
for i in range(len(predicted_out)):
    if predicted_out[i] == real_out[i]:
        count += 1

accuratezza = count/len(Y_effettivo)
print(count, len(Y_effettivo), accuratezza)



