import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme(style="darkgrid")


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_diff(x):
    return sigmoid(x)*(1-sigmoid(x))

def ReLU(x):
    if x<=0:
        return 0
    else:
        return x

def ReLU_diff(x):
    if x<=0:
        return 0
    else:
        return 1
    
def tanh(x):
    return 1-2/(np.exp(2*x)+1)

def tanh_diff(x):
    return 1-tanh(x)**2

def Leak_relu(x):
    if x<=0:
        return 0.01*x
    else:
        return x
    
def silu(x):
    return x*sigmoid(x)

def silu_diff(x):
    return sigmoid(x)*(1+x-x*sigmoid(x))

def draw_func(xs, ys):
    for input_x, output_y in zip(xs, ys):
        data = pd.DataFrame({"input_x":input_x,
                            "output_y":output_y})
        sns.lineplot(data=data, x="input_x", y="output_y")
    plt.show()





if __name__=="__main__":
    func_name = silu_diff
    x1 = np.linspace(-10, 5, num=500)
    y1 = [func_name(item) for item in x1]
    # x2 = list(np.linspace(0, 10, num=50))[1:]
    # y2 = [ReLU_diff(item) for item in x2]
    draw_func([x1], [y1])