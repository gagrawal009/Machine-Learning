import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data
#X = np.array([0.02, 0.12, 0.19, 0.27, 0.42, 0.51, 0.64, 0.84, 0.88, 0.99])
#t = np.array([0.05, 0.87, 0.94, 0.92, 0.54, -0.11, -0.78, -0.79, -0.89, -0.04])

data = pd.read_csv("housing.csv")
X = data.iloc[1:,0:3].values
Y = data.iloc[1:,3].values
t=Y


#PHI = X.reshape (X.shape[0],1)
PHI = X
print ("PHI = ")
print (PHI)


# Bayesian Linear Regression
alpha = 0.1 # assume
beta = 10  # assume
Sigma = np.linalg.inv(alpha * np.identity(PHI.shape[1]) + beta * np.dot(PHI.T, PHI))
mu = beta * np.dot(Sigma, np.dot(PHI.T, t))
print ("mu (Bayesian linear regression) = ")
print (mu)


#loss
output = (np.dot(PHI, mu) - t)
output = np.square (output)
print ("bayesian linear loss", (np.sum(output))/PHI.shape[0])






