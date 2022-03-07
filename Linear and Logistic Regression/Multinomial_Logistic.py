import numpy as np
import pandas as pd
import decimal

train_data = pd.read_csv("mnist_train.csv")
test_data = pd.read_csv("mnist_test.csv")

def preprocessor(data):
    X = data.iloc[:,1:785].values
    Y = data.iloc[:,0].values
    j=0
    Ycap = np.zeros((Y.shape[0],10))
    for i in range(Y.shape[0]):
        ycap = np.zeros(10)
        ycap[Y[i]] = 1
        Ycap[i] = ycap
    return X, Ycap


def softmax(x):
    B = np.exp(x)
    for i in range (len(B)):
        if (B[i].any()<0.0000000000000000000001):
            B[i]=0
    p = (np.sum(np.exp(x), axis=1)).reshape(x.shape[0],1)
    return B / p


def loss(T, O):
    return np.sum((-T*np.log(O)))


def grad (X, T, O):
    ans = np.dot(X.T,(O-T))
    return ans

#initialise data
"""
X = np.array([[1, 0.1, 0.5],
              [1, 1.1, 2.3],
              [1, -1.1, -2.3],
              [1, -1.5, -2.5]])

Y = np.array([[1,0,0],
              [0,1,0],
              [0,0,1],
              [0,0,1]])"""

X, Y = preprocessor(train_data)
X = X/255
X_test = test_data.values/255

alpha = 0.01  #learning rate
num_it = 100  #number of iterations

#initialise W
W = np.random.randn (X.shape[1], Y.shape[1])*0.0001

for i in range (num_it):
    Z = np.dot (X,W)
    O = softmax(Z)
    k = grad (X,Y,O)
    
    W = W - alpha*k
    W = np.clip(W,-0.5,0.5)
    
    
target = np.argmax(Y, axis = 1)
print ("Given labels of train data: ", target)

training_output = np.argmax ((softmax(np.dot (X,W))), axis = 1)
print ("training_output of train data: ", training_output)

result = np.argmax ((softmax(np.dot (X_test,W))), axis = 1)
print ("training_output of test data: ", result)


"""
temp =0
for i in range (X.shape[0]):
    if ans[i]==y_temp[i]:
        temp+=1

print((temp)/X.shape[0])
    
""" 
    
    
    


