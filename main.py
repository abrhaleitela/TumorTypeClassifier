# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:13:32 2019

@author: abrye
"""

import numpy as np
from matplotlib import pyplot
from sklearn.neural_network import MLPRegressor
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
import csv

# Neural network params
HIDDEN_SIZE = [4,]
# "identity", "tanh", "logistic", "relu"
ACTIVATION = 'logistic'
LEARNING_RATE = 0.01

file = open('tcga-pancan-hiseqlabels.csv', 'r')
data = list(csv.reader(file, delimiter=','))

"""======  DATA PREPROCESSING   ======="""

#prepare a dictionary classes that holds 'sample-id'->'crossponding class'
classes = {}
for row in data:   
    classes[row[0]] = row[1]
    
# =============================================================================
""" Prepare X as [[sample number 1],[sample number 2], .... [sample number n]]
    And Y as [[class of sample 1],[class of sample 2], .... [class of sample n]] 
"""
# =============================================================================

data_set = np.genfromtxt("expressions.txt", dtype = None)
X = []
Y = []
for row in data_set:
    y = classes[row[0].decode("utf-8")]
    Y.append(y)
    i = 1
    x = []  
    while i < len(row):
        x.append(float(row[i]))
        i = i + 1
    X.append(x)           
X = np.array(X)
print(X.shape)

""" Shuffle two arrays (X and Y) in same way..."""
X,Y = shuffle(X,Y)

""" Standardizing of input features """
stdsc = StandardScaler()
X = stdsc.fit_transform(X)

""" Split train and test sets[25% as test set] """
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
 
""" Build sequential NN """
classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(100, activation='relu', kernel_initializer='random_normal', input_dim = 2257))
#Second  Hidden Layer
classifier.add(Dense(100, activation='relu', kernel_initializer='random_normal'))
#Third  Hidden Layer
classifier.add(Dense(50, activation='relu', kernel_initializer='random_normal'))
#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

#Compiling the neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

#Fitting the data to the training dataset
classifier.fit(X_train,Y_train, batch_size=10, epochs=100)

eval_model=classifier.evaluate(X_train, Y_train)
print(eval_model)








