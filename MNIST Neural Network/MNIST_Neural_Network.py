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

X, y = preprocessor(train_data)
X = X/255
X_test = test_data.values/255

np.random.seed(1)
def initialise (layer_dims):
    initial_params = {}
    L=len (layer_dims)
    
    for i in range (L-1):
        initial_params ['W' + str(i+1)] = np.random.randn(layer_dims[i+1], layer_dims[i])*0.1
        initial_params ['b' + str(i+1)] = np.zeros ((layer_dims[i+1], 1))
    return initial_params

def sigmoid(x):
    return 1/(1+np.exp(x))

def relu(x):
    if x>0:
        return x
    else:
        return np.zeros(x.shape)

def softmax(x): 
    B = np.exp(x)
    for i in range (len(B)):
        if (B[i].any()<0.0000000000000000000001):
            B[i]=0
    return B / np.sum(np.exp(x), axis=0)


def forward_activation(inputs, W, b, function):
    output =0
    temp2 =0
    if function == "sigmoid":
        net = np.dot (W, inputs) + b
        temp = (inputs, W, b)
        output = sigmoid(net)
        temp2 = (net)
    
    elif function == "relu":
        net = np.dot (W,inputs) + b
        temp = (inputs, W, b)
        output = relu(net)
        temp2 = (net)
        
    elif function == "softmax":
        net = np.dot (W, inputs) + b
        temp = (inputs, W, b)
        output = softmax(net)
        temp2 = (net)
        
    return output, temp2, temp


def forward_prop (input, initial_params):
    caches = []
    caches_net =[]
    A =input.T
    L = len(initial_params)//2
    
    for i in range(1,L):
        input_here = A
        A, temp2, temp = forward_activation (input_here, initial_params['W' + str(i)], initial_params['b' + str(i)], function = "sigmoid")
        caches.append (temp)
        caches_net.append (temp2)
        
    output, temp2, temp = forward_activation (A, initial_params['W' + str(L)], initial_params['b' + str(L)], function = "softmax")
    caches.append (temp)
    caches_net.append (temp2)

    return output, caches, caches_net

#using cross entropy loss function
def loss(training_output, output):
    
    m = output.shape[0]
    loss_value = -(np.dot(output.T, np.log(training_output).T) + np.dot((1-output.T), np.log(1-training_output).T))/m
    return np.sum(abs(loss_value))

def sigmoid_backward (dcost, value):
    diff = sigmoid(value)*(1-sigmoid(value))
    return dcost*diff

def relu_backward (dcost, value):
    if value>0:
        diff = 1
    else :
        diff = 0
    return dcost*diff

def backward(inputs):
    value = softmax(inputs)
    gradient = np.zeros((10,len(inputs[0])))
    for k in range(len(inputs[0])):
        for i in range(len(value)):
            for j in range(len(value)):
                if i == j:
                    gradient[i][k] = value[i][k] * (1-value[i][k])
                else: 
                    gradient[i][k] = -value[i][k]*value[j][k]
    return gradient

def softmax_backward (dcost, value):
    diff = softmax(value)
    return dcost*diff

def linear_backward(dcost, cache):
    input, W, b = cache
    m= input.shape[1]
    dW = np.dot(dcost, input.T)/m
    db = np.sum(dcost.T, axis=0)/m

    dinput = np.dot (W.T, dcost)
    return dinput, dW, db

def backward_activation (doutput, caches, caches_net, function):
    temp2 = caches_net
    temp1 = caches
    if function == "relu":
        dnet = relu_backward (doutput, temp2)
        dinput, dW, db = linear_backward (dnet, temp1)
    
    elif function == "sigmoid":
        dnet = sigmoid_backward (doutput, temp2)
        dinput, dW, db = linear_backward (dnet, temp1)
    elif function == "softmax":
        dnet = softmax_backward (doutput, temp2)
        dinput, dW, db = linear_backward (dnet, temp1)
        
    return dinput, dW, db



def back_prop (training_output, output, caches, caches_net):
    gradient = {}
    L=len(caches)
    
    AL = training_output
    m=training_output.shape[1]
    Y=output
    Y = Y.reshape(training_output.shape)
    
    dAL =  - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    temp = caches[L-1]
    temp2 = caches_net[L-1]
    gradient["dA" + str(L-1)], gradient["dW" + str(L)], gradient["db" + str(L)] = backward_activation(dAL, temp, temp2, function="softmax")

    for l in reversed(range(L-1)):
        temp = caches[l]
        temp2 = caches_net[l]
        gradient["dA" + str(l)], gradient["dW" + str(l + 1)], gradient["db" + str(l + 1)]  = backward_activation(gradient["dA" + str(l + 1)], temp, temp2, function="sigmoid")
        
    return gradient


def update_params (initial_params, gradient, alpha):
    L = len(initial_params) // 2 
    
    for l in range(L):
        initial_params["W" + str(l+1)] = initial_params["W" + str(l+1)] - alpha* gradient["dW" + str(l+1)]
        initial_params["b" + str(l+1)] = initial_params["b" + str(l+1)] - alpha* gradient["db" + str(l+1)].reshape(gradient["db" + str(l+1)].shape[0],1)
    return initial_params

def predict(input):
    
    if(len(input.shape) == 1):
        A = input.reshape(input.shape[0],1)    
    else:
        A = input.T
        
    L = len(initial_params)//2
    
    for i in range(1,L):
        input_here = A
        A, temp2, temp = forward_activation (input_here, initial_params['W' + str(i)], initial_params['b' + str(i)], function = "sigmoid")
        
    output, temp2, temp = forward_activation (A, initial_params['W' + str(L)], initial_params['b' + str(L)], function = "softmax")
        
    return np.argmax(output, axis = 0)


layer_dims = [784, 50, 40, 10]
initial_params = initialise (layer_dims)

for i in range (100):
    training_output, caches, caches_net = forward_prop(X, initial_params)
    loss_value_k = loss(training_output, y)
    
    gradient = back_prop (training_output, y, caches, caches_net)
    alpha = 0.006
    initial_params = update_params (initial_params, gradient, alpha)
    print(loss_value_k)


ans = predict(X)
y_temp = np.argmax(y, axis = 1)
temp =0
for i in range (X.shape[0]):
    if ans[i]==y_temp[i]:
        temp+=1

print((temp)/X.shape[0])


y_test = predict(X_test/255)
pd.DataFrame(y_test).to_csv("test_output.csv")











    
