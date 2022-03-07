
#from google.colab import files
#uploaded = files.upload()

import pandas as pd
import random
import numpy as np
from numpy import linalg
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import copy

dataset = pd.read_csv('boston.csv')
X = dataset.iloc[:,0:13].values.astype(float)
y = dataset.iloc[:,13].values.astype(float)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

#Split dataset into k folds
def cross_validation_split(X, y, folds=3):
  
  random.seed(5)
  X_split = list()
  X_copy = list(X)
  y_split = list()
  y_copy = list(y)
  num_elements  = int(len(X) / folds)
  
  for i in range(folds):
    foldX = np.empty((num_elements,X.shape[1]))
    foldy = np.empty((num_elements,1))
    insert=0
    k=0
    
    while insert < num_elements:
      index = random.randrange(len(X_copy))
      
      foldX[k]= X_copy.pop(index)
      foldy[k] = y_copy.pop(index)
      insert=insert+1
      k=k+1
    
    X_split.append(foldX)
    y_split.append(foldy)
    
  return X_split, y_split


def linear_kernel (x1, x2):
  return np.dot(x1, x2)

def polynomial_kernel (x, y, p=3):
  return (1+np.dot(x,y.T))**p

def gaussian_kernel(x, y, sigma=5.0):
  return np.exp(-linalg.norm(x-y)**2/(2*(sigma**2)))

#def sigmoid_kernel(x, y):

def getG(m,C):
  cos1 = np.eye(2*m)
  for i in range (m):
    for j in range(m):
      if (i==j):
        cos1[i][j]=-1
  
  cos2 = np.eye(2*m)
  for i in range (m, 2*m):
    for j in range(m, 2*m):
      if (i==j):
        cos2[i][j]=-1

  res = np.vstack((cos1, cos2))  
  return res

def optimize (X,y,C,epsilon, kernel=gaussian_kernel):
  m,n = X.shape
  y = y.reshape(-1,1) * 1.

  #Converting into cvxopt format
  K = np.zeros((m,m))
  
  for i in range(m):
    for j in range(m):
      K[i,j] = kernel(X[i], X[j])
  
  #K=kernel(X,X)
  

  M = np.hstack((K,K))
  M = np.vstack((M,M))
  P = cvxopt_matrix(M)
  q = cvxopt_matrix((np.hstack(((np.ones(m)*epsilon)-y.T , (-np.ones(m)*epsilon)-y.T))).T)
  G = cvxopt_matrix(getG(m,C))
  h = cvxopt_matrix(np.hstack((np.zeros(2*m), np.ones(2*m) * C)))
  A = cvxopt_matrix(np.ones((1,2*m)))
  b = cvxopt_matrix(np.zeros(1))
  
  cvxopt_solvers.options['show_progress'] = False
  cvxopt_solvers.options['abstol'] = 1e-10
  cvxopt_solvers.options['reltol'] = 1e-10
  cvxopt_solvers.options['feastol'] = 1e-10
  
  #Run solver
  sol = cvxopt_solvers.qp(P, q, G, h, A, b)
  alphas = np.array(sol['x'])

  answer = np.empty((2*m,1))
  
  for i in range (m):
    answer[i] = alphas[i]

  for i in range (m, 2*m):
    answer[i] = -alphas[i]

  for i in range (2*m):
    if (answer[i]<0.0001):
      answer[i] = 0
    if ((abs(alphas[i]))<0.0001):
      alphas[i] = 0

 
  for j in range(m):
    if (alphas[j]>0 and alphas[j]<C*1.0):
      k=0
      for i in range(m):
        k = k+ (alphas[i]+alphas[m+i])*kernel(X[i], X[j])
      bstar = (y[j]-k+epsilon)
      break
 
  return answer, alphas, bstar

def find_mse (y, y_pred, m_test):
  test_error=0.0
  for i in range(m_test):
    #print (y_test[i], y_pred[i])
    test_error = test_error + (y[i]-y_pred[i])**2
  test_error = np.sqrt (test_error/(m_test*1.0))
  return test_error

def find_r2score (y, y_pred, m_test):
  r2_score=0
  y = y.reshape(-1,1)
  y_pred = y_pred.reshape(-1,1)
 
  num = np.sum((y-y_pred)**2)
  y_bar = np.mean(y)
  den = np.sum((y-y_bar)**2)
  r2_score = 1-(num/den)
  
  return r2_score

def test_function (X, y, X_test, y_test, alphas, bstar, kernel=gaussian_kernel):

  m,n = X.shape
  m_test, n_test = X_test.shape
  factor = np.empty((m,1))
  dim_alpha = (int) (alphas.shape[0]/2)
  for i in range(dim_alpha):
    factor[i] = alphas[i]+alphas[dim_alpha+i]
  
  y_pred = np.empty(m_test)
  sum=0
  for j in range (m_test):
    sum=0
    for i in range(m):
      sum = sum + factor[i] * kernel(X[i], X_test[j])
    y_pred[j] = sum+bstar

  mse_test = find_mse(y_test, y_pred, m_test)
  r2_score_test = find_r2score (y_test, y_pred, m_test)


  y_pred_train = np.empty(m)
  sum=0
  for j in range (m):
    sum=0
    for i in range(m):
      sum = sum + factor[i] * kernel(X[i], X[j])
    y_pred_train[j] = sum+bstar

  mse_train = find_mse(y, y_pred_train, m)
  r2_score_train = find_r2score (y, y_pred_train, m)

  return mse_train, r2_score_train, mse_test, r2_score_test
  


C=500
folds = 3
epsilon = 0.29

X_split, y_split = cross_validation_split(X,y,folds)
  
mse_store_train = []
scores_train=[]
mse_store_test = []
scores_test=[]

for j in range(folds):
    
  X_train = np.empty((X_split[0].shape[0]*(folds-1),X_split[0].shape[1]))
  y_train = np.empty(y_split[0].shape[0]*(folds-1))
  X_test = np.empty((X_split[0].shape[0],X_split[0].shape[1]))
  y_test = np.empty(y_split[0].shape[0])
    
  X_test= X_split[j]
  y_test= y_split[j]

  count=0
  i=(j+1)%folds
  k=0
  while count<(folds-1):
    for l in range (X_split[i].shape[0]):
      X_train[k] = X_split[i][l]
      y_train[k] = y_split[i][l]
      k=k+1

    count = count+1
    i = (i+1)%folds
    
  answer, alphas, bstars = optimize(X_train, y_train, C, epsilon, kernel)
  mse_train, r2_score_train, mse_test, r2_score_test=test_function(X_train,y_train, X_test, y_test, alphas,bstars)
    
  mse_store_train.append(mse_train)
  scores_train.append (r2_score_train)
  #print ("mse_train: ",mse_train, " r2_score_train: ", r2_score_train)
  
  mse_store_test.append(mse_test)
  scores_test.append (r2_score_test)
  #print ("mse_test: ",mse_test[0], " r2_score_test: ", r2_score_test)


print ("mean(mse_train): " ,np.mean(mse_store_train), " mean(r2_score_train): ", np.mean(scores_train))
print ("mean(mse_test): " ,np.mean(mse_store_test), " mean(r2_score_test): ", np.mean(scores_test))
