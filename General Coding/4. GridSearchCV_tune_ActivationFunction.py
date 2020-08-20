##-------------------------- Example 4. Tune Activation Function ---------------------------------

#The activation function controls the non-linearity of individual neurons and when to fire.

#Generally, the rectifier activation function is the most popular, but it used to be the sigmoid
#and the tanh functions and these functions may still be more suitable for different problems.

#Import libraries
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV

# Model function
def model_func(activation='relu'):
    model=Sequential()
    model.add(Dense(12,activation=activation,input_dim=8))
    model.add(Dense(8,activation=activation))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    return model

#Random seed
seed=7
np.random.seed(seed)

# Data read
df=np.loadtxt('pima-indians-diabetes.txt',delimiter=',')

# Data split
X=df[:,0:8]
y=df[:,8]

# Model create
model=KerasClassifier(build_fn=model_func,epochs=100,batch_size=10,verbose=0)

# GridSearchCV for activation function
activation=['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
param_grid=dict(activation=activation)
grid=GridSearchCV(estimator=model,param_grid=param_grid,n_jobs=-1,cv=3)
grid_results=grid.fit(X,y)

#Results
with open('Results_GridSearchCV_tune_Activation.txt','a') as f:
    print('Best %f using %s' % (grid_results.best_score_,grid_results.best_params_),file=f)
means=grid_results.cv_results_['mean_test_score']
stds=grid_results.cv_results_['std_test_score']
params=grid_results.cv_results_['params']

for mean,stdev,param in zip(means,stds,params):
    with open('Results_GridSearchCV_tune_Activation.txt','a') as f:
        print('%f (%f) with: %r' % (mean,stdev,param),file=f)
