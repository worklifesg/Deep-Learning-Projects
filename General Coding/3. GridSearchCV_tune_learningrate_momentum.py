##-------------------------- Example3. Tune Learning Rate and Momentum ---------------------------------

# In this example, SGD (Stochastic Gradient Descent) is used to tune its learning rate and momentum parameters
# Learning rate controls how much to update the weight at the end of each batch 
# Momentum controls how much to let the previous update influence the current weight update.

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD

from sklearn.model_selection import GridSearchCV

# Defining model function
def model_func():
    model=Sequential()
    model.add(Dense(12,activation='relu',input_dim=8))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(1,activation='relu'))

    optimizer=SGD(learning_rate=learn_rate,momentum=momentum) # defining optimizer and its parameters that needed to be tuned
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    return model

# Random seed
seed=7
np.random.seed(seed)

#Data read
df=np.loadtxt('pima-indians-diabetes.csv',delimiter=',')

# Split the data
X=df[:,0:8]
y=df[:,8]

# Create model
model=KerasClassifier(build_fn=model_func,epochs=100,batch_size=10,verbose=0)

#GridSearchCV for LR and momentum
learn_rate=[0.001,0.01,0.1,0.2,0.3]
momentum=[0.0,0.2,0.4,0.6,0.8,0.9]

param_grid=dict(learn_rate=learn_rate,momentum=momentum)
grid=GridSearchCV(estimator=model,param_grid=param_grid,n_jobs=-1,cv=3)
grid_results=grid.fit(X,y)

#Results
with open('Results_GridSearchCV_tune_LR_momentum','a') as f:
    print('Best %f using %s' % (grid_results.best_score_,grid_results.best_params_))
means=grid_results.cv_results_['mean_test_score']
stds=grid_results.cv_results_['std_test_score']
params=grid_results.cv_results_['params']

for mean,stdev,param in zip(means,stds,params):
    with open('Results_GridSearchCV_tune_LR_momentum.txt','a') as f:
        print('%f (%f) with: %r' % (mean,stdev,param),file=f)
