#-------------------------------------------------------------------------
# AUTHOR: Joseph Luna
# FILENAME: perceptron.py
# SPECIFICATION: Create a perceptron model for OCR
# FOR: CS 4210- Assignment #4
# TIME SPENT: ~30 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

#print(np.shape(X_training))
#print(X_training[0])

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

best = dict()
best['P'] = (0, 0, 0, 0)
best['MLP'] = (0, 0, 0, 0)

for lr in n: #iterates over n

    for doShuffle in r: #iterates over r

        #iterates over both algorithms
        algos = ['P', 'MLP']

        for algo in algos: #iterates over the algorithms

            #Create a Neural Network classifier
            if algo == 'P':
               clf = Perceptron(eta0=lr, shuffle=doShuffle, max_iter=1000)    #use those hyperparameters: eta0 = learning rate, shuffle = shuffle the training data, max_iter=1000
            else:
               clf = MLPClassifier(activation='logistic', learning_rate_init=lr, hidden_layer_sizes=100, shuffle=doShuffle, max_iter=1000) #use those hyperparameters: activation='logistic', learning_rate_init = learning rate, hidden_layer_sizes = number of neurons in the ith hidden layer,
            #                          shuffle = shuffle the training data, max_iter=1000
            #-->add your Pyhton code here

            #Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            #make the classifier prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously with zip() Example:
            total = 0
            correct = 0
            for (x_testSample, y_testSample) in zip(X_test, y_test):
                prediction = clf.predict([x_testSample])
                total += 1
                if (prediction[0] == y_testSample):
                    correct += 1

            #to make a prediction do: clf.predict([x_testSample])
            #--> add your Python code here

            #check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy
            #and print it together with the network hyperparameters
            #Example: "Highest Perceptron accuracy so far: 0.88, Parameters: learning rate=0.01, shuffle=True"
            #Example: "Highest MLP accuracy so far: 0.90, Parameters: learning rate=0.02, shuffle=False"
            if best[algo][3] < (correct/total):
                best[algo] = (lr, doShuffle, algo, (correct/total))
                print(f"Highest {'Perceptron' if algo == 'P' else 'MLP'} so far {best[algo][3]}, Parameters: learning rate={best[algo][0]}, shuffle={best[algo][1]}")




