# In this example, we will pass more arguments to fit() to further customize the construction of the model
# Here we use grid search to evaluate different configurations for our neural neteork model and report on combination that provides best estimated performance.



#-------------------------- Example1: Tune Batch size and number of Epochs ---------------------------------
#The batch size in iterative gradient descent is the number of patterns shown to the network before the weights are updated.
#The number of epochs is the number of times that the entire training dataset is shown to the network during training. 


# Import Keras and scikit-learn libraries
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV

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

model=KerasClassifier(build_fn=model_func,verbose=0)

# Grid search - epochs,batch size, optimizer

epochs=[10,50,100]
batches=[10,20,40,60,80,100]

para_grid=dict(epochs=epochs,batch_size=batches)
grid=GridSearchCV(estimator=model,param_grid=para_grid,n_jobs=-1, cv=3)
grid_result=grid.fit(X,y)

## Results
with open('Results_GridSearchCV_tune_batchepoch.txt','a') as f:
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_),file=f)

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    with open('Results_GridSearchCV_tune_batchepoch.txt','a') as f:
        print("%f (%f) with: %r" % (mean, stdev, param),file=f)
