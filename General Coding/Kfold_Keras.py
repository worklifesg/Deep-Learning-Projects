# K-fold cross validation technique is one of the best machine learning model evaluation technique.
# - provides robust performance estimation of model and unseen data
# - training dataset - k subsets, turns training model and one held out for validation performance
# - process is repeated until all subsets are taken for validation performance set.
## However, it is computationaly expensive for deep learning projects delaing with large dataset

## In similar handy fashion, StratifiesKFold class (scikit-learn) is used to split training dataset into k folds.
# In the end, avg and std of all performances is calculated to see optimium accuracy performance.

# Importing keras and scikit-learn libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import StratifiedKFold

# Random seed for reproductivity
seed=7
np.random.seed(seed)

# Dataset
df=np.loadtxt('pima-indians-diabetes.txt',delimiter=',')

# Split data
X=df[:,0:8] # inputs
y=df[:,8] #output

# Taking 12 neurons for 1st hidden layer and 8 neurons for 2nd hidden layer with one output and using 'relu' as activation function
def model_func():
    model = Sequential()
    model.add(Dense(12, input_dim=8 , activation= 'relu' ))
    model.add(Dense(8, activation= 'relu' ))
    model.add(Dense(1, activation= 'sigmoid' ))

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model

#using k=10, we define 10 fold cross validation test harness
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
cvscores=[] #creating an empty array to store the accuracy results for each fold

for train,test in kfold.split(X,y):
    model=model_func()
    model.fit(X[train],y[train],epochs=150,batch_size=10,verbose=0) # train the dataset
    scores=model.evaluate(X[test],y[test],verbose=0) # evaulating test dataset
    with open('kfold_keras.txt','a') as f:
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100),file=f)
    cvscores.append(scores[1] * 100)

with open('kfold_keras.txt','a') as f:
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)),file=f)
