
import numpy as np


'''
Review of gradient decent algorithm and stocastic gradient decent and mini batch gradient decent

'''



def gredient(a):
    # calculate the deriative here
    return a'



def greadient_decent(epoch, gradient, learning_rate, start):
    weight_init = start

    for i in range (epoch):
        diff = gradient(weight_init) * learning_rate 
        vec -=  diff
    return vec



import torch
def SGD(
        gredient, x, y, start, epoch, learning_rate=2e-6, bacth_size = 1,
        tolenrance = 1e-3,  random_seed = 7 
):
    
    vec = start
    x, y = torch.Tensor(x), torch.Tensor(y)
    ss = x.shape[0]

    xy = torch.cat(x.view(ss, -1), y.view(ss, -1))
    lr = torch.Tensor(learning_rate)

    for i in range(epoch):

        for bt in range(0, ss, bacth_size):
            stop = bt + bacth_size
            x_batch, y_batch = xy[bt:stop, :-1], xy[bt:stop, -1:]

        derivative  = gredient(x_batch, y_batch)
        change = derivative * lr

        if torch.all(abs(change)) <= tolenrance:
            break

        vec -= change
    
    return vec if vec.shape else vec.item()






