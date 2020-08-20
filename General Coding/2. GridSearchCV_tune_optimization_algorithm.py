##-------------------------- Example2: Tune Optimization Algorithm ---------------------------------

# To tune the optimization algorithm used to train the network, each with default parameters.

import numpy as np 
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# Create Sqeuential model with hidden layers
def model_fun(optimizer='adam'):
    model=Sequential()
    model.add(Dense(12,input_dim=8,activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model

# Random seed
seed=7
np.random.seed(seed)

# Data read
df=np.loadtxt('pima-indians-diabetes.txt',delimiter=',')
# Split data
X=df[:,0:8]
y=df[:,8]
# Create model
model=KerasClassifier(build_fn=model_fun,epochs=100,batch_size=10,verbose=0)

# GridSearch parameters
optimizer=['SGD','RMSprop','Adagrad','Adadelta','Adam','Adamax','Nadam']
param_grid=dict(optimizer=optimizer)
grid=GridSearchCV(estimator=model,param_grid=param_grid,n_jobs=-1,cv=3)
grid_results=grid.fit(X,y)

#Results
with open('Results_GridSearchCV_tune_optimizer.txt', 'a') as f:
    print('Best: %f using %s' % (grid_results.best_score_,grid_results.best_params_), file=f)

means=grid_results.cv_results_['mean_test_score']
stds=grid_results.cv_results_['std_test_score']
params=grid_results.cv_results_['params']

for mean,stdev,param in zip(means,stds,params):
    with open('Results_GridSearchCV_tune_batchepoch.txt','a') as f:
        print('%f (%f) with: %r' % (mean,stdev,param),file=f)
