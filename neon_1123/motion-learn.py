"numerical gradient function"
"input: function f, funciton f's input x"
def numerical_gradient(f, x):
    h = 1e-4
    "zeros like: same shape 0"
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        "calculate f(x+h), h is a very small amount"
        x[idx] = tmp_val + h 
        fxh1 = f(x)
        
        "calculate f(x-h)"
        x[idx] = tmp_val - h 
        fxh2 = f(x)
        
        grad[idx] = (fxh1-fxh2)/(2*h)
        
    return grad


"gradient descent"
"input: function f, initial f's input init_x, learning rate lr, times of interation step_num"
def gradient_descent(f, init_x, lr=0.01, step_num = 100):
    x = init_x
    "for every step:"
    for i in range(step_num):
        "get gradient for each x"
        grad = numerical_gradient(f, x)
        
        x -= lr*grad
    
    print("100% of one flag")
    return x