# In general, Keras provide two convenient ways to evaluate deep learning algorithms:

## Automatic Verification Dataset - automatically taken by keras and validation split is done in fit()
# Manual Verification dataset - using scikit-learn library, train_test_split and mentioning validation data in fit()

## In this example, we will use pima indians diabetes dataset available in repository

import numpy as np
import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense

#Initialize the random number generator with a fixed seed value.
#This is so that you can run the same code again and again and get the same result.

seed=7
np.random.seed(seed)

## Loading the data
df=np.loadtxt('pima-indians-diabetes.csv',delimiter=',')

#Split the data 
X=df[:,0:8] # input
y=df[:,8] # output

# Taking 12 neurons for 1st hidden layer and 8 neurons for 2nd hidden layer with one output and using 'relu' as activation function
def model_func():
    model = Sequential()
    model.add(Dense(12, input_dim=8 , activation= 'relu' ))
    model.add(Dense(8, activation= 'relu' ))
    model.add(Dense(1, activation= 'sigmoid' ))

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model

model_automatic=model_func()
model_automatic.fit(X,y,validation_split=0.33,epochs=150,batch_size=10)

## For manual verification, we need to use scikit-learn library and integrate with keras

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=seed)

model_manual=model_func()
model_manual.fit(X,y,validation_data=(X_test,y_test),epochs=150,batch_size=10) 
