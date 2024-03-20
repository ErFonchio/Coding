import numpy as np
import math
import random as rn

class Map:
    def __init__(self):
        self.count = 0
        self.count_dataset = 0
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
                #print(string)
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
    
    def mappingDataset(self, dataset):
        colonne = dataset.columns
        for elem in colonne:
            print(elem)
            self.dataset_dictionary[:, elem] = {}
            print(dataset.loc[elem])

    
    def MSE(self, a, b, axis):
        return (np.square(a - b)).mean(axis)
    def RMSE(self, a, b, axis):
        return math.sqrt((np.square(a - b)).mean(axis))
    def MAE(self, a, b, axis):
        return abs(a-b).mean(axis)
    
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
        
        #print("[LDT]: entering\t", self)
        #print("[LDT]: examples\t", examples)
        #print("[LDT]: attributes\t", attributes)
        #print("[LDT]: parent_examples\t", parent_examples)

        same_classification = self.SameClassification(examples)
        #print("[LDS]: SameClassification\t", same_classification)
        
        if len(examples.loc[:, examples.columns != self.outputstring]) == 0:
            #print("[LDS]: len(examples) == 0")
            return self.PluralityValue(parent_examples)
        elif same_classification is not False:
            #print("[LDS]: same_classification is not False ", same_classification)
            return same_classification
        elif len(attributes) == 0: 
            #print("[LDS]: len(attributes) == 0")
            return self.PluralityValue(examples)
        else:
            bestattribute = self.Importance(attributes, examples)
            self.label = bestattribute #Seleziono l'attributo migliore
            #print("valori pazzerelli: ", column_values)
            for value in self.Values1(bestattribute, column_values):
                #print("[LDT]: value ", value, " of attribute ", self.label)
                #print("[LDT]: examples before\n\n", examples)
                remainingexamples = self.Examples(bestattribute, examples, value)
                #print("Remainingexamples: ", remainingexamples)
                tree = DecisionTree(self.outputstring, self.positive, self.negative)
                
                attributes_left = self.PopListValue(attributes.copy(), bestattribute)
                #print("[LDS]: attributes_left ", attributes_left)
                
                subtree = tree.LearnDecisionTreeAux_(remainingexamples.loc[:, remainingexamples.columns != bestattribute], attributes_left, examples, column_values)
                #print("ok")
                self.nodes.append((value, subtree))
                #print(self.nodes)
        return self
        
        
    def Importance(self, attributes, examples): 
        max = -1
        ret = None
        for a in attributes:
            #print("[Importance]: Testing importance of ", a, " on\n\n", examples)
            loc = self.Gain(examples, a)
            #print("[Importance] attribute ", a, "has Gain ", loc)
            if loc > max:
                max = loc
                ret = a
        #print("[Importance]: most important attribute is\t", ret)
        return ret
        
    def B(self, q):
        if q == 1 or q == 0:
            #print("[B]: q = 1 or q = 0")
            return 0
        return -(q*math.log2(q)+(1-q)*math.log2(1-q))
    
    def Remainder(self, examples, attribute, p, n):
        sum = 0
        #print("[Remainder]: values of ", attribute, " are: ", self.Values(attribute, examples))
        for v in self.Values2(attribute, examples):
            #print("[Remainder]: examples:\n\n", examples)
            #print("[Remainder]: calculating for value: ", v)
            #print("[Remainder]: calculating for attribute: ", attribute)
            if type(v) is not str and math.isnan(float(v)):
                pk = len(examples.loc[(examples[self.outputstring] == self.positive) & (examples[attribute].isnull())])
                #print("[Remainder] ROBO PK:\n\n", examples.copy(deep=True).loc[(examples[self.outputstring] == self.positive) & ((examples[attribute] == v) | (examples[attribute].isnull()))])
                nk = len(examples.loc[(examples[self.outputstring] == self.negative) & (examples[attribute].isnull())])
                #print("[Remainder] ROBO NK:\n\n", examples.copy(deep=True).loc[(examples[self.outputstring] == self.negative) & ((examples[attribute] == v) | (examples[attribute].isnull()))])
            else:
                pk = len(examples.loc[(examples[self.outputstring] == self.positive) & (examples[attribute] == v)])
                nk = len(examples.loc[(examples[self.outputstring] == self.negative) & ((examples[attribute] == v))])

            #print("[Remainder]: pk and nk ", pk, nk, " of attribute ", attribute, "and value ", v)
            partial = (pk+nk)*self.B(pk/(pk+nk))
            #print("[Remainder] Partial of attribute ", attribute, " and value ", v, " is: ", partial)
            sum += partial
        return (1/(p+n))*sum
    
    def Gain(self, examples, attribute):
        p = len(examples.loc[examples[self.outputstring] == self.positive])
        n = len(examples.loc[examples[self.outputstring] == self.negative])
        b = self.B(p/(p+n))
        r = self.Remainder(examples, attribute, p, n)
        #print("[Gain]: Gain of attribute ", attribute, " has B: ", b, " and Remainder: ", r)
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
        #print("[CreateDictionary]: After examples:\n", examples)
        d = {}
        for i in range(len(examples)):
            if d.get(examples[i]) is None:
                d[examples[i]] = 1
            else:
                d[examples[i]] += 1
        #print("[CreateDictionary]: Dictionary:", d)
        return d

    def Values1(self, attribute, dictionary): 
        #print(dictionary)
        #print("[Values]: values of ", attribute, " are: ", dictionary.get(attribute))
        return list(dictionary.get(attribute))
    
    def Values2(self, attribute, examples):
        return examples[attribute].unique()

    def Examples(self, attribute, examples, value):
        exs = examples.loc[examples[attribute] == value]
        #print("[Examples] examples: ", exs)
        return exs
    def PopListValue(self, lista, value):
        #print("[PopListValue]: lista ", lista, " da cui levo ", value)
        if value not in lista:
            #print("[PopListValue]: value ", value, " not in list")
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
        #print(input)
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

            
    
    
    
