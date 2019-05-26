# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:13:32 2019

@author: abrye
"""

import numpy as np
from matplotlib import pyplot
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from keras import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
import csv
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


LEARNING_RATE = 1e-6
EPOCHS = 40
BATCH_SIZE = 10

def dataPreprocessing():  
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
    Y2 = []
    for row in data_set:
        y = classes[row[0].decode("utf-8")]
        Y2.append(int(y))
        if int(y) == 1:
            y = [1,0,0,0,0]
        elif int(y) == 2:
            y = [0,1,0,0,0]
        elif int(y) == 3:
            y = [0,0,1,0,0]
        elif int(y) == 4:
            y = [0,0,0,1,0]
        else:
            y = [0,0,0,0,1]
        Y.append(y)
        i = 1
        x = []  
        while i < len(row):
            x.append(float(row[i]))
            i = i + 1
        X.append(x)           
    X = np.array(X)
    return X,Y,Y2
    
def prepareDataset(X , Y):
    """ Shuffle two arrays (X and Y) in same way..."""
    X,Y = shuffle(X,Y)
    """ Standardizing of input features """
    stdsc = StandardScaler()
    X = stdsc.fit_transform(X)
    """ Split train and test sets[25% as test set] """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    return(X_train, X_test, Y_train, Y_test)

def geneSelectionForClassifcation(X, start, end):
    return(X[: , start:end])
    
def sequentialKeras(x_train, y_train, x_test, y_test):
    input_dim = np.size(x_train,1)
    """ Build sequential NN """
    model = Sequential()
    """ First Hidden Layer """
    model.add(Dense(10, activation='relu', kernel_initializer='random_normal', input_dim = input_dim))
    #model.add(Dropout(0.5))
    """ Output Layer """
    model.add(Dense(5, activation='softmax'))
    """ Early stopping  """
    callbacks = [EarlyStopping(monitor='val_loss', patience = 2),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
    """ Compiling the neural network """
    model.compile(optimizer ='adam',loss='categorical_crossentropy', metrics =['accuracy'])
    
    
    """ Fitting the data to the training dataset  """
    model.fit(np.array(x_train),np.array(y_train),validation_split = 0.30, callbacks = callbacks, batch_size = BATCH_SIZE, epochs = EPOCHS)
    
    loss, acc = model.evaluate(np.array(x_test), np.array(y_test))
    print("Accuracy in test set: " + str(acc))
    print("Loss in test set: " + str(loss))


def gaussianNB(x_train, y_train,x_test,y_test):    
    gnb = GaussianNB()
    pred = gnb.fit(x_train, y_train).predict(x_test)
    print("Naive-Bayes accuracy : ",accuracy_score(y_test, pred, normalize = True))
    
def SVM(x_train, y_train,x_test,y_test):
    svc_model = LinearSVC(random_state=0)
    pred = svc_model.fit(x_train, y_train).predict(x_test)
    print("LinearSVC accuracy : ",accuracy_score(y_test, pred, normalize = True))
    
def KN(x_train, y_train,x_test,y_test):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x_train, y_train)
    pred = neigh.predict(x_test)
    print ("KNeighbors accuracy score : ",accuracy_score(y_test, pred))    

def visualize(X,y):
    pca = PCA(n_components=2) 
    x = pca.fit_transform(X, y = y)
    pyplot.scatter(x[:,0], x[:,1] , c = y)
    pyplot.show()

def T_SNE(X,y):
    x = TSNE(n_components=2).fit_transform(X)
    pyplot.scatter(x[:,0], x[:,1] , c = y)
    pyplot.title("2D representation of the given data set")
    pyplot.show()
    
def plot(x,y,value):
    pyplot.scatter(x, y , c = y)
    pyplot.plot([value[1],value[1]],[0,6])
    pyplot.title("IF GENE '" +str(value[0]) + "' EXPRESSION >= " + str(value[1]) + " THEN THE SAMPLE IS FROM TUMER TYPE '" +str(value[2])+"'")
    pyplot.ylabel("Tumer types labeled as 1 to 5")
    pyplot.xlabel("Sample data's just from gene " +str(value[0])+' expression')
    pyplot.show()
    
def visualizeOneGene(values):
     for value  in values:
         # Remove genes from the data set X starting from index i to j(two paramaters given in the function below)
         new_X = geneSelectionForClassifcation(X, value[0],value[0]+1)
         plot(new_X,Y2,value)
           
""" TASK NUMBER 1 """
print("")
print("")
print("")
print('"""    TASK NUMBER 1   """')
print("")
print("")
print("")
X,Y,Y2 = dataPreprocessing()
X_train, X_test, Y_train, Y_test = prepareDataset(X , Y)
sequentialKeras(X_train, Y_train, X_test,Y_test)
T_SNE(X,Y2)
""" TASK NUMBER 1 USING DIFFERENT ALGORITHMS(they all provide accuracy of ~100%) """
# =============================================================================
# X_train, X_test, Y_train, Y_test = prepareDataset(X , Y2)
# gaussianNB(X_train, Y_train, X_test, Y_test)
# SVM(X_train, Y_train,X_test,Y_test)
# KN(X_train, Y_train,X_test,Y_test)
# =============================================================================

"""    TASK NUMBER 2   """
print("")
print("")
print("")
print('"""    TASK NUMBER 2   """')
values =[[395,9,5],[1508,9,4],[1154,13,1],[1816,5.5,2]]
visualizeOneGene(values)
#visualize(X,Y2)


