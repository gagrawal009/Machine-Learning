
#from google.colab import files
#uploaded = files.upload()

import pandas as pd
dataset = pd.read_csv('boston.csv')
X = dataset.iloc[:,0:13].values.astype(float)
y = dataset.iloc[:,13].values.astype(float)

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np

sc_X = StandardScaler()
X = sc_X.fit_transform(X)

n_folds=11

scores=[]
mse_store=[]
kfold = KFold(n_folds, shuffle=True, random_state=1)
regressor = SVR(kernel, epsilon = 1, C=500, degree=2) #linear, poly, rbf, sigmoid
epsilon=0.1
for train_ix, test_ix in kfold.split(X):
  trainX, trainY, testX, testY = X[train_ix], y[train_ix], X[test_ix], y[test_ix]
  regressor.fit(trainX, trainY)
  y_pred = regressor.predict(testX)
  mse = mean_squared_error(testY.reshape(-1,1), y_pred.reshape(-1,1))
  mse_store.append(np.sqrt(mse))
  #print ("mse: ",np.sqrt(mse), " r2_score: ", regressor.score(testX, testY))
  scores.append(regressor.score(testX, testY))
  
print ("mean(mse): " ,np.mean(mse_store), " mean(r2_score): ", np.mean(scores))

