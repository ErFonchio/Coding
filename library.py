import numpy as np
import math
import random as rn
import pandas as pd

class Map:
    def __init__(self):
        self.count = 0
        self.count_dataset = {}
        self.d = {}
        self.dataset_dictionary = {}
        self.matrice = None
        self.dataset = None
        len = None
    
    def mappingElement(self, string):
        if type(string) is str:
            if self.d.get(string) is not None:
                return self.d.get(string)
            else:
                self.count = self.count+1
                self.d[string] = self.count
                return self.count
        return float(string)
    
    def mappingMatrix(self, matrice):
        self.matrice = None
        self.matrice = np.array(matrice)
        len = self.matrice.shape
        for g in range(len[0]):
            for j in range(len[1]):
                self.matrice[g][j] = self.mappingElement(self.matrice[g][j])
        return np.float64(self.matrice.copy())
    
    def mappingElementDataset(self, value, colonna):
        if type(value) is str:
            if self.dataset_dictionary[colonna].get(value) is not None:
                return self.dataset_dictionary[colonna].get(value)
            else:
                if self.count_dataset.get(colonna) is None:
                    self.count_dataset[colonna] = 0
                    self.dataset_dictionary[colonna][value] = self.count_dataset[colonna]
                    return self.count_dataset[colonna]
                self.count_dataset[colonna] += 1
                self.dataset_dictionary[colonna][value] = self.count_dataset[colonna]
                return self.count_dataset[colonna]
        return value
    
    def mappingDataset(self, dataset):
        self.dataset = dataset
        colonne = dataset.columns
        length = len(dataset)
        for column in colonne:
            self.dataset_dictionary[column] = {}
            for index in range(length):
                dataset.at[index, column] = self.mappingElementDataset(dataset.at[index, column], column)
        return self.dataset

    def MSE(self, a, b):
        return np.mean((np.square(a - b)))
    def RMSE(self, a, b):
        return math.sqrt(np.mean(np.square(a - b)))
    def MAE(self, a, b):
        return np.mean(abs(a-b))
    

class DecisionTree:
    def __init__(self, outputstring, positive, negative): 
        self.outputstring = outputstring
        self.positive = positive
        self.negative = negative
        self.nodes = []
        self.label = None
        self.root = False

    def LearnDecisionTree(self, examples, attributes, parent_examples, column_values):
        self.root = True
        self.LearnDecisionTreeAux_(examples, attributes, parent_examples, column_values)

    def LearnDecisionTreeAux_(self, examples, attributes, parent_examples, column_values):
        same_classification = self.SameClassification(examples)    
        
        if len(examples.loc[:, examples.columns != self.outputstring]) == 0:
            return self.PluralityValue(parent_examples)
        elif same_classification is not False:
            return same_classification
        elif len(attributes) == 0: 
            return self.PluralityValue(examples)
        else:
            bestattribute = self.Importance(attributes, examples)
            self.label = bestattribute #Seleziono l'attributo migliore
            
            for value in self.Values1(bestattribute, column_values):
                remainingexamples = self.Examples(bestattribute, examples, value)
                tree = DecisionTree(self.outputstring, self.positive, self.negative)
                attributes_left = self.PopListValue(attributes.copy(), bestattribute)                
                subtree = tree.LearnDecisionTreeAux_(remainingexamples.loc[:, remainingexamples.columns != bestattribute], attributes_left, examples, column_values)
                self.nodes.append((value, subtree))
        return self
        
    def Importance(self, attributes, examples): 
        max = -1
        ret = None
        for a in attributes:
            loc = self.Gain(examples, a)
            if loc > max:
                max = loc
                ret = a
        return ret
        
    def B(self, q):
        if q == 1 or q == 0:
            return 0
        return -(q*math.log2(q)+(1-q)*math.log2(1-q))
    
    def Remainder(self, examples, attribute, p, n):
        sum = 0
        
        for v in self.Values2(attribute, examples):
            if type(v) is not str and math.isnan(float(v)):
                pk = len(examples.loc[(examples[self.outputstring] == self.positive) & (examples[attribute].isnull())])
                nk = len(examples.loc[(examples[self.outputstring] == self.negative) & (examples[attribute].isnull())])
            else:
                pk = len(examples.loc[(examples[self.outputstring] == self.positive) & (examples[attribute] == v)])
                nk = len(examples.loc[(examples[self.outputstring] == self.negative) & ((examples[attribute] == v))])
            
            partial = (pk+nk)*self.B(pk/(pk+nk))
            sum += partial
        return (1/(p+n))*sum
    
    def Gain(self, examples, attribute):
        p = len(examples.loc[examples[self.outputstring] == self.positive])
        n = len(examples.loc[examples[self.outputstring] == self.negative])
        b = self.B(p/(p+n))
        r = self.Remainder(examples, attribute, p, n)
        return (b-r)
        
    def PluralityValue(self, parent_examples): 
        '''Selects the most common ouput value among a set of examples, breaking ties randomly'''
        value, max = [], 0
        d = self.CreateDictionary(parent_examples)
        for key in d.keys():
            if d.get(key) > max:
                max = d.get(key)
                value = []
                value.append(key)
            elif d.get(key) == max:
                '''se ci sono più massimi li metto tutti in una lista dalla quale ne
                    selezionerò uno randomicamente
                '''
                value.append(key)
        return rn.choice(value)
        
    def SameClassification(self, examples): 
        if (len(examples) == 0):
            return False
        d = self.CreateDictionary(examples)
        if (len(d.keys()) == 1):
            return list(d.keys())[0]
        return False
    def CreateDictionary(self, examples):
        examples = examples[self.outputstring].tolist()
        d = {}
        for i in range(len(examples)):
            if d.get(examples[i]) is None:
                d[examples[i]] = 1
            else:
                d[examples[i]] += 1
        return d

    def Values1(self, attribute, dictionary): 
        return list(dictionary.get(attribute))
    
    def Values2(self, attribute, examples):
        return examples[attribute].unique()

    def Examples(self, attribute, examples, value):
        exs = examples.loc[examples[attribute] == value]
        return exs
    def PopListValue(self, lista, value):
        if value not in lista:
            return None
        lista.remove(value)
        return lista 
    def PrintDecisionTree(self, count):
        print(" "*count, self.label)
        for elem in self.nodes:
            if type(elem[1]) is not DecisionTree:
                print("    "*(count+1), elem[0], " --> ", elem[1])
            else:
                elem[1].PrintDecisionTree(count+1)

    def Prediction(self, input):
        for i in range(len(self.nodes)):
            x = input[self.label].values[0]
            if type(self.nodes[i][0]) is not str and math.isnan(float(self.nodes[i][0])):
                if type(x) is not str and math.isnan(float(x)):
                    if type(self.nodes[i][1]) is DecisionTree:
                        return self.nodes[i][1].Prediction(input)
                    return self.nodes[i][1]
            elif self.nodes[i][0] == x:
                if type(self.nodes[i][1]) is DecisionTree:
                    return self.nodes[i][1].Prediction(input)
                return self.nodes[i][1]
        return None

class GradientDescent:
    def __init__(self, learningrate, epochs, parameters_length, output_string): 
        self.learningrate = learningrate
        self.epochs = epochs
        self.parameters_length = parameters_length
        self.parameters = None
        self.output_string = output_string

    def SGD(self, training_set):
        self.parameters = np.ones(self.parameters_length)
        for i in range(self.epochs):

            for g in range(1):

                #training_set = training_set.sample(frac=1) #shuffle sample in the training set
                input = training_set.loc[g:g, training_set.columns != self.output_string]
                output = training_set.loc[g:g, training_set.columns == self.output_string]
                gradient = self.gradient(input, output)
                self.SGD_aux(gradient)
            
            if (not (i%100)):
                print(self.parameters)
        return self.parameters
                

    def SGD_aux(self, gradient):
        for f in range(self.parameters_length):
            self.parameters[f] = self.parameters[f]-(self.learningrate*gradient[f])

    def gradient(self, input, output): 
        prediction = self.prediction(input)
        error = prediction - output
        return np.dot(input.T, error)

    def prediction(self, input):
        #input già sezionato
        return np.dot(input, self.parameters)

    def MSE(self, prediction, output):
        return np.mean((output - prediction)**2)

