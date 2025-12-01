import numpy as np

# NETWORK STRUCTURE
#
# i    
#      h
# i         o
#      h
# i
#

# DIMENSIONS
# 3  2  1

def ReLU(x):
    return np.maximum(0,x)

def deriv_ReLU(x):
    return np.where(x > 0, 1, 0)

def init_params():
    #first sets of weights and biases, 2x3 and 2x1 
    W1 = np.random.rand(2,3) - 0.5
    B1 = np.random.rand(2,1) - 0.5
    
    #second set of weights and biases, 1x2 and 1x1 
    W2 = np.random.rand(1,2) - 0.5
    B2 = np.random.rand(1,1) - 0.5
    
    return W1, B1, W2, B2
    
def forward_pass(X, W1, B1, W2, B2):
    A1 = W1.dot(X) + B1   #2x1
    A1F = ReLU(A1)        #2x1
    FF = W2.dot(A1) + B2   #1x1 
    
    return A1, A1F, FF

def backward_pass(A1, A1F, FF, X, target):
    dFF = 2 * (FF - target)        #1x1 
    dW2 =  dFF.dot(A1F.T)         #1x2     
    dB2 = dFF               #1x1
    dA1 = deriv_ReLU(A1) * dW2.T.dot(dFF)   #2x1
    dW1 = dA1.dot(X.T)        #2x3               
    dB1 = dA1          #2x1
    
    return dW1, dB1, dW2, dB2

def update_params(dW1, dB1, dW2, dB2, W1, B1, W2, B2, learningRate):
    W1 = W1 - learningRate*dW1
    B1 = B1 - learningRate*dB1
    W2 = W2 - learningRate*dW2
    B2 = B2 - learningRate*dB2
    
    return W1, B1, W2, B2
    

X = np.array([[-1, 1, 0]]).T    
TARGET = 4
LEARNINGRATE = 0.01
EPOCHS = 300
countCorrect, countWrong = 0, 0

for i in range(100):
    W1, B1, W2, B2 = init_params()
    for i in range (EPOCHS):
        A1, A1F, FF = forward_pass(X, W1, B1, W2, B2)
        dW1, dB1, dW2, dB2 = backward_pass(A1, A1F, FF, X, TARGET)
        W1, B1, W2, B2 = update_params(dW1, dB1, dW2, dB2, W1, B1, W2, B2, LEARNINGRATE)
        cost = (TARGET - FF) ** 2
    if FF > 1.99: 
        countCorrect +=1
    else:
        countWrong +=1
        
print(f"correct: {countCorrect/100} ({countCorrect}/100)")
print(f'wrong {countWrong/100} ({countWrong}/100)')

# to test one run

# W1, B1, W2, B2 = init_params()
# for i in range (EPOCHS):
#         A1, A1F, FF = forward_pass(X, W1, B1, W2, B2)
#         dW1, dB1, dW2, dB2 = backward_pass(A1, A1F, FF, X, TARGET)
#         W1, B1, W2, B2 = update_params(dW1, dB1, dW2, dB2, W1, B1, W2, B2, LEARNINGRATE)
#         cost = (TARGET - FF) ** 2
#         if i%10 == 0:
#            print(f"cost: {cost}")
#            print(f"final: {FF}")
    