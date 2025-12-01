import numpy as np

# NETWORK STRUCTURE
#
# i    
#      h
# i         o
#      h
# i
#

x1, x2, x3, w11, w21, w12, w22, w13, w23, wi11, wi12, bi1, b1, b2, target, j1, j2 = [0] * 17
final, w11grad, w21grad, w12grad, w22grad, w13grad, w23grad, wi11grad, wi12grad, b1grad, b2grad, bi1grad = [0] * 12

def init_params():
    global x1, x2, x3, w11, w21, w12, w22, w13, w23, wi11, wi12, bi1, b1, b2, target, j1, j2, final, w11grad, w21grad, w12grad, w22grad, w13grad, w23grad, wi11grad, wi12grad, b1grad, b2grad, bi1grad
    
    x1, x2, x3 = -1, 1, 0       #inputs
    w11, w21, w12, w22, w13, w23 = np.random.rand() - 0.5, np.random.rand() - 0.5, np.random.rand() - 0.5, np.random.rand() - 0.5, np.random.rand() - 0.5, np.random.rand() - 0.5       #weights and biases b/w hidden layer and inputs
    b1, b2  = np.random.rand() - 0.5 , np.random.rand() - 0.5 
    wi11, wi12 = np.random.rand() - 0.5, np.random.rand() - 0.5         #weights and biases b/w output and hidden layer   
    bi1 = np.random.rand() - 0.5 
    target = 4
    j1, j2, final, w11grad, w21grad, w12grad, w22grad, w13grad, w23grad, wi11grad, wi12grad, b1grad, b2grad, bi1grad = [0] * 14
    
def ReLU(x):
    return x if x > 0 else 0

def deriv_ReLU(x):
    return 1 if x > 0 else 0
    
def forward_pass():
    global x1, x2, x3, w11, w21, w12, w22, w13, w23, wi11, wi12, bi1, b1,b2, target, j1, j2, final, w11grad, w21grad, w12grad, w22grad, w13grad, w23grad, wi11grad, wi12grad, b1grad, b2grad, bi1grad
    
    #the neurons of the hidden layer
    j1 = ReLU(x1*w11 + x2*w12 + x3*w13 + b1)
    j2 = ReLU(x1*w21 + x2*w22 + x3*w23 + b2)
    
    #the single neuron of the output layer
    final = j1*wi11 + j2*wi12 + bi1 

def backward_pass():
       global x1, x2, x3, w11, w21, w12, w22, w13, w23, wi11, wi12, bi1, target, j1, j2, b1, b2, final, w11grad, w21grad, w12grad, w22grad, w13grad, w23grad, wi11grad, wi12grad, b1grad, b2grad, bi1grad
       
       #gradient of loss with respect to last neuron
       final_grad = 2 * (final -  target) 
       #derivative of loss w/respect to final where loss uses MSE: (final-target)^2
       
       #gradients of wi11 and wi22 and bi1
       wi11grad = j1 * final_grad 
       # dLoss     dLoss     dFinal
       # -----  =  -----  x  ----- 
       # dwi11     dFinal    dwi11
       wi12grad = j2 * final_grad
       
       #gradient of bi1
       bi1grad = final_grad 
       
       #gradients of w11, w12 ... w23
       w11grad = deriv_ReLU(x1*w11 + x2*w12 + x3*w13 + b1) * x1  * wi11 * final_grad

       # THE MATH IS VERY SIMPLE!!! JUST CHAIN RULES AND DERIVATIVES!!!
       # dLoss     dLoss     dFinal     dj1
       # -----  =  -----  x  -----  x  -----
       # dw11      dFinal     dj1       dw11

       #final = j1*wi11 + j2*wi12 + bi1, so
       # dFinal
       # -----  =  wi11
       #  dj1

       # so the equation becomes: 
       # dLoss                               dj1
       # -----  =  final_grad  x  wi11   x  -----
       # dw11                                dw11

       #dj1 = ReLU(x1*w11 + x2*w12 + x3*w13 + b1), so 
       #  dj1                        dj1
       # -----  =  derivReLu(j1) *  -----     , and
       #  dw11                       dw11 

       #  dj1 
       # -----  =  x1
       #  dw11

       # so the equation is now:
       # dLoss     dLoss     dFinal     dj1
       # -----  =  -----  x  -----  x  -----   =  final_grad * wi11 * derivRelu(j1) * xi
       # dw11      dFinal     dj1       dw11
       w21grad = deriv_ReLU(x1*w21 + x2*w22 + x3*w23 + b2) * x1  * wi12 * final_grad
       w12grad = deriv_ReLU(x1*w11 + x2*w12 + x3*w13 + b1) * x2  * wi11 * final_grad
       w22grad = deriv_ReLU(x1*w21 + x2*w22 + x3*w23 + b2) * x2  * wi12 * final_grad
       w13grad = deriv_ReLU(x1*w11 + x2*w12 + x3*w13 + b1) * x3  * wi11 * final_grad
       w23grad = deriv_ReLU(x1*w21 + x2*w22 + x3*w23 + b2) * x3  * wi12 * final_grad

       # Gradients for b1 and b2
       b1grad = deriv_ReLU(x1*w11 + x2*w12 + x3*w13 + b1) * wi11 * final_grad
       b2grad = deriv_ReLU(x1*w21 + x2*w22 + x3*w23 + b2) * wi12 * final_grad
       
def update_params():
    global x1, x2, x3, w11, w21, w12, w22, w13, w23, wi11, wi12, bi1, b1, b2, target, j1, j2, final, w11grad, w21grad, w12grad, w22grad, w13grad, w23grad, wi11grad, wi12grad, b1grad, b2grad, bi1grad
    lr = 0.01
    w11 += -w11grad * lr
    w21 += -w21grad * lr
    w12 += -w12grad * lr
    w22 += -w22grad * lr
    w13 += -w13grad * lr
    w23 += -w23grad * lr
    b1 += -b1grad * lr
    b2 += -b2grad * lr
    wi11 += -wi11grad * lr
    wi12 += -wi12grad * lr
    bi1 += -bi1grad * lr

correct = 0
wrong = 0

for i in range(100):
    init_params()
    for i in range(500):
        forward_pass()
        backward_pass()
        update_params()
    if final > 1.99: 
        correct +=1
    else:
        wrong +=1
        
print(f"correct: {correct/100} ({correct}/100)")
print(f'wrong {wrong/100} ({wrong}/100)')

# to test one run

# init_params()
# for i in range(300):
#     forward_pass()
#     backward_pass()
#     update_params()
#     cost = (target - final)**2
#     if i%10 == 0:
#         print(f"cost: {cost}")
#         print(f"final: {final}")


    
    
    
            
       
       
       
