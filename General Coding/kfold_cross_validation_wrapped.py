# The idea here is to evaluate models using resampling methods like k-fold cross validation alongside with convenient wrapper
# such as KerasClassifier

# Importing keras and scikit-learn libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

# Taking 12 neurons for 1st hidden layer and 8 neurons for 2nd hidden layer with one output and using 'relu' as activation function
def model_func():
    model = Sequential()
    model.add(Dense(12, input_dim=8 , activation= 'relu' ))
    model.add(Dense(8, activation= 'relu' ))
    model.add(Dense(1, activation= 'sigmoid' ))

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model

# Random seed for reproductivity
seed=7
np.random.seed(seed)
# Dataset
df=np.loadtxt('pima-indians-diabetes.txt',delimiter=',')
# Split data
X=df[:,0:8] # inputs
y=df[:,8] #output

# Create model
model = KerasClassifier(build_fn=model_func, epochs=150,batch_size=10,verbose=0)

#Evaluate using 10 fold cross validation
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
results=cross_val_score(model,X,y,cv=kfold)

with open('kfold_keras_wrapped.txt','a') as f:
    print('The mean score is ', results.mean(),file=f)

## You can see that when the Keras model is wrapped that estimating model accuracy can be
## greatly streamlined, compared to the manual enumeration of cross validation folds performed
## in kfold_cross_validation.py
