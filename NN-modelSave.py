import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from math import sqrt
from numpy.random import seed
from keras.regularizers import l1
import random
from keras.models import model_from_json
import os
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, make_scorer, roc_auc_score, balanced_accuracy_score, confusion_matrix, precision_score, recall_score, fbeta_score
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from keras.wrappers.scikit_learn import KerasClassifier
from numpy import mean, std
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler
from imblearn.pipeline import Pipeline, make_pipeline
import random

seed = 6
np.random.seed(seed)

dataframe = pandas.read_csv("FCTnoST2.csv", header=0)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X_orig = dataset[:,1:18].astype(float)
y_orig = dataset[:,18:19].astype(float)

print(X_orig)

#Function 1: All functions may change in modification 
def generate_data1(training_data, number_samples):
    generated_set = np.empty((0,12))
    sampling_set1= [1,-1,0]
    sampling_set2= [1,2,3] 
    
    for i in range(number_samples):
        row_index = i % len(training_data)
        new_row = training_data[row_index,:].copy().reshape(1,12)
        for j in range(4):
            change_endors = np.random.choice(sampling_set1,1)
            change_severity= np.random.choice(sampling_set2, 1)*change_endors
            new_row[0,j] = new_row[0,j] + change_endors
            if new_row[0, j] > 5:
                new_row[0,j] = 5
            if new_row[0,j] <= 0:
                new_row[0,j+4] = 0
            else: 
                new_row[0,j+4] = new_row[0,j+4] + change_severity
                
            if new_row[0,j+4] > new_row[0,j]*3 :
                new_row[0, j+4] = (new_row[0,j].copy())*3
                
            if new_row[0,j+4] < new_row[0,j] :
                new_row[0, j+4] = (new_row[0,j].copy())
            
        new_row[new_row < 0] = 0

        generated_set = np.vstack((generated_set, new_row.flatten()))
    
    return generated_set


def generate_data(data, number_samples): 
    generated_set = np.empty((0, 18))
    print(data)
    ran_sample = data[np.random.randint(data.shape[0], size=1), :]
    print(ran_sample)

    for i in range(number_samples):
        ran_sample = data[np.random.randint(data.shape[0], size=1), :]
        new_row = np.copy(ran_sample)
        #Depending upon the function change the rate of behavior in test and control
        #6 is attention, 7 is escape, 8 is tangible

        if ran_sample[0][3] == 1:
            new_row[0][14] = (np.random.uniform(low=0.83, high=1.17, size=(1,))) * ran_sample[0][14] # 83 to 100 percent random number between IOA error for aggression
            #new_row[0][15] = (np.random.uniform(low=0.83, high=1.17, size=(1,))) * ran_sample[0][15]#random number between IOA and error for aggression
            generated_set = np.vstack((generated_set, new_row))
        elif ran_sample[0][4]==1:
            new_row[0][14] = (np.random.uniform(low=0.83, high=1.17, size=(1,))) * ran_sample[0][14]# 83 to 100 random number bettwen IOA and error for disruption
            #new_row[0][15] = (np.random.uniform(low=0.83, high=1.17, size=(1,))) * ran_sample[0][15]#random number between IOA and error for diruption
            generated_set = np.vstack((generated_set, new_row))
        elif ran_sample[0][2] == 1:
            new_row[0][14] = (np.random.uniform(low=0.85, high=1.15, size=(1,))) * ran_sample[0][14] # 85 to 100 percent random number between IOA error for SIB
           # new_row[0][15] = (np.random.uniform(low=0.85, high=1.15, size=(1,))) * ran_sample[0][15] #random number between IOA and error for SIB
            generated_set = np.vstack((generated_set, new_row))
        elif ran_sample[0][5] == 1:
            new_row[0][14] = (np.random.uniform(low=0.88, high=1.13, size=(1,))) * ran_sample[0][14]#88 to 99.8 random number bettwen IOA and error for other
            #new_row[0][15] = (np.random.uniform(low=0.88, high=1.13, size=(1,))) * ran_sample[0][15]#random number bettwen IOA and error for other
            generated_set = np.vstack((generated_set, new_row))
    return generated_set



################################################################################################################
def NeuralNet():
    model = Sequential()
    model.add(Dense(16, activation='linear', input_dim=X_orig.shape[1]))
    model.add(Activation('relu'))
    model.add(Dense(y_orig.shape[1], activation='sigmoid'))
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=[tf.keras.metrics.AUC()])
    return model



cv = LeaveOneOut()
y_true, y_pred = list(), list()
for train_ix, test_ix in cv.split(X_orig):
    # split data
    X_train, X_test = X_orig[train_ix, :], X_orig[test_ix, :]
    y_train, y_test = y_orig[train_ix], y_orig[test_ix]
   
   #Generate New Samples 
    num_samp = 1000
    data_send = np.append(X_train, y_train, axis = 1)
    gen_data = generate_data(data_send, num_samp)
    retrieved_data = gen_data
    y_additional = retrieved_data[:, -1]
    X_additional = retrieved_data[:, :-1]

    X_train = np.vstack((X_train, X_additional))
    y_additional =y_additional.reshape((-1, 1))
    print(y_additional)
    print(y_train)
    y_train = np.vstack((y_train, y_additional))
    #Fit model
    model = KerasClassifier(NeuralNet, epochs=200, batch_size = 15)
    model.fit(X_train, y_train)
    # evaluate model
    yhat = model.predict(X_test)
    # store
    y_true.append(y_test[0])
    y_pred.append(yhat[0])
print(accuracy_score(y_true, y_pred))
print(balanced_accuracy_score(y_true, y_pred))

