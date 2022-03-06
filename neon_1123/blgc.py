import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
import random
from cycler import cycler
from sklearn.decomposition import PCA
import matplotlib.collections as collections
import time
import logging

"constants"
ppc = np.load('w_synergy.npy')
mean = np.load('mean.npy')
link_average = np.load('linklength.npy')

columnname = ['ankle', 'knee', 'hip', 'shoulder', 'neck', 'elbow']
columnnamexyz = np.load('columnnamexyz.npy')

filenumber = 10
mknumber =11

"episodes: M1 to M10 motion sepoints list"
episodes = [[[0,300], [301, 613], [614,854],[855,1155]],[[1156,1336],[1337, 1577],[1578, 1806],[1807, 2047]],
            [[2048, 2276], [2277, 2517], [2518, 2758], [2759, 3058]], [[3059, 3311],[3312, 3612],[3613, 3913],[3914,4154]],
            [[4155, 4875], [4876, 5416]],[[5417, 5777],[5778, 6258]],
            [[6259, 6919], [6920, 7460], [7461, 7941]],[[7942, 8241],[8242, 8663]],[[8664,9084],[9085,9590],[9591,10070],[10071,10431]],[[10432,10768],[10769,11129],[11130,11550]]]

"DICTIONARY xat: x position of marker i (key) at column k (value)"
xat = {
    1:8,2:11,3:14,4:17,5:20,6:23,7:26,8:29,9:32,10:2,11:5
}
"FUNCTION yat: y position of marker i at column k (xat+1)"
def yat(key):
    return xat[key]+1


"dfT: input dataframe, must be angles with headers of jointname"
"output: xyz dataframe (same format as Mn.csv), can be used to calculate error"
"DIFFERENT FROM rebuild_from.py series, this only process a single dataframe"
"NOT lists which includes dataframes"
def blgc_model(dfT):
    
    newlist = []
    for i in range(len(dfT)):

        rbtslice = np.zeros(35)
        rbtslice[xat[1]] = 0
        rbtslice[yat(1)] = 0
        "calculate x2"
        rbtslice[xat[2]] = rbtslice[xat[1]] + link_average[0]*math.cos(dfT.iloc[i]['ankle'])
        rbtslice[yat(2)] = rbtslice[yat(1)] + link_average[0]*math.sin(dfT.iloc[i]['ankle'])
        "calculate x3"
        ang = dfT.iloc[i]['knee']+dfT.iloc[i]['ankle']
        rbtslice[xat[3]] = rbtslice[xat[2]] + link_average[1]*math.cos(ang)
        rbtslice[yat(3)] = rbtslice[yat(2)] + link_average[1]*math.sin(ang)

        "m4 is decided by a fixed link, more close to the original with this kind of bias"

        "method1"
        xh = rbtslice[xat[3]] + 0.65*link_average[1]*math.cos(ang)
        yh = rbtslice[yat(3)] + 0.65*link_average[1]*math.sin(ang)

        phi = ang+dfT.iloc[i]['hip']+ math.pi*0.27

        rbtslice[xat[4]] = xh + 0.65*link_average[1]*math.cos(phi)
        rbtslice[yat(4)] = yh + 0.65*link_average[1]*math.sin(phi)


        ang = ang+dfT.iloc[i]['hip']


        rbtslice[xat[5]] = rbtslice[xat[4]] + link_average[3]*math.cos(ang)
        rbtslice[yat(5)] = rbtslice[yat(4)] + link_average[3]*math.sin(ang)

        "m6 has a little bit bias towards m5 ,-15 deg ~ -18 deg"
        rbtslice[xat[6]] = rbtslice[xat[5]] + link_average[4]*math.cos(ang - np.deg2rad(10))
        rbtslice[yat(6)] = rbtslice[yat(5)] + link_average[4]*math.sin(ang - np.deg2rad(10))

        "firstly, m8"

        ang = ang+dfT.iloc[i]['neck']

        rbtslice[xat[8]] = rbtslice[xat[6]] + link_average[5]*math.cos(ang)
        rbtslice[yat(8)] = rbtslice[yat(6)] + link_average[5]*math.sin(ang)

        "m7 has a little bit bias towards m8, +"
        rbtslice[xat[7]] = rbtslice[xat[8]] + link_average[6]*math.cos(ang + np.deg2rad(5))
        rbtslice[yat(7)] = rbtslice[yat(8)] + link_average[6]*math.sin(ang + np.deg2rad(5))

        "firstly m10 "
        ang = ang-dfT.iloc[i]['neck']+dfT.iloc[i]['shoulder']

        rbtslice[xat[10]] = rbtslice[xat[6]] + link_average[7]*math.cos(ang)
        rbtslice[yat(10)] = rbtslice[yat(6)] + link_average[7]*math.sin(ang)

        "m9 has a little bit bias towards m10"
        rbtslice[xat[9]] = rbtslice[xat[10]] + link_average[8]*math.cos(ang - np.deg2rad(6))
        rbtslice[yat(9)] = rbtslice[yat(10)] + link_average[8]*math.sin(ang - np.deg2rad(6))

        ang = ang + dfT.iloc[i]['elbow']
        rbtslice[xat[11]] = rbtslice[xat[10]] + link_average[9]*math.cos(ang)
        rbtslice[yat(11)] = rbtslice[yat(10)] + link_average[9]*math.sin(ang)
        newlist.append(rbtslice)
        
    "make dataframe"
    dfreturn = pd.DataFrame(data=newlist, columns=columnnamexyz)
    return dfreturn

def blgc_model_slice(dfT):
    
    newlist = []


    rbtslice = np.zeros(35)
    rbtslice[xat[1]] = 0
    rbtslice[yat(1)] = 0
    "calculate x2"
    rbtslice[xat[2]] = rbtslice[xat[1]] + link_average[0]*math.cos(dfT['ankle'])
    rbtslice[yat(2)] = rbtslice[yat(1)] + link_average[0]*math.sin(dfT['ankle'])
    "calculate x3"
    ang = dfT['knee']+dfT['ankle']
    rbtslice[xat[3]] = rbtslice[xat[2]] + link_average[1]*math.cos(ang)
    rbtslice[yat(3)] = rbtslice[yat(2)] + link_average[1]*math.sin(ang)

    "m4 is decided by a fixed link, more close to the original with this kind of bias"

    "method1"
    xh = rbtslice[xat[3]] + 0.65*link_average[1]*math.cos(ang)
    yh = rbtslice[yat(3)] + 0.65*link_average[1]*math.sin(ang)

    phi = ang+dfT['hip']+ math.pi*0.27

    rbtslice[xat[4]] = xh + 0.65*link_average[1]*math.cos(phi)
    rbtslice[yat(4)] = yh + 0.65*link_average[1]*math.sin(phi)


    ang = ang+dfT['hip']


    rbtslice[xat[5]] = rbtslice[xat[4]] + link_average[3]*math.cos(ang)
    rbtslice[yat(5)] = rbtslice[yat(4)] + link_average[3]*math.sin(ang)

    "m6 has a little bit bias towards m5 ,-15 deg ~ -18 deg"
    rbtslice[xat[6]] = rbtslice[xat[5]] + link_average[4]*math.cos(ang - np.deg2rad(10))
    rbtslice[yat(6)] = rbtslice[yat(5)] + link_average[4]*math.sin(ang - np.deg2rad(10))

    "firstly, m8"

    ang = ang+dfT['neck']

    rbtslice[xat[8]] = rbtslice[xat[6]] + link_average[5]*math.cos(ang)
    rbtslice[yat(8)] = rbtslice[yat(6)] + link_average[5]*math.sin(ang)

    "m7 has a little bit bias towards m8, +"
    rbtslice[xat[7]] = rbtslice[xat[8]] + link_average[6]*math.cos(ang + np.deg2rad(5))
    rbtslice[yat(7)] = rbtslice[yat(8)] + link_average[6]*math.sin(ang + np.deg2rad(5))

    "firstly m10 "
    ang = ang-dfT['neck']+dfT['shoulder']

    rbtslice[xat[10]] = rbtslice[xat[6]] + link_average[7]*math.cos(ang)
    rbtslice[yat(10)] = rbtslice[yat(6)] + link_average[7]*math.sin(ang)

    "m9 has a little bit bias towards m10"
    rbtslice[xat[9]] = rbtslice[xat[10]] + link_average[8]*math.cos(ang - np.deg2rad(6))
    rbtslice[yat(9)] = rbtslice[yat(10)] + link_average[8]*math.sin(ang - np.deg2rad(6))

    ang = ang + dfT['elbow']
    rbtslice[xat[11]] = rbtslice[xat[10]] + link_average[9]*math.cos(ang)
    rbtslice[yat(11)] = rbtslice[yat(10)] + link_average[9]*math.sin(ang)
    newlist.append(rbtslice)
        
    "make dataframe"
    dfreturn = pd.DataFrame(data=newlist, columns=columnnamexyz)
    return dfreturn

"return np array"
def pick_up_episode(Mi, epn, file):
    st = episodes[Mi-1][epn][0]
    ed = episodes[Mi-1][epn][1]
    vvst = episodes[Mi-1][0][0]
    if(type(file[0]) == np.ndarray):
        "if array, to dataframe"
        return pd.DataFrame(file[Mi-1][st-vvst:ed-vvst], columns = columnname)
    if(type(file[0]) == pd.core.frame.DataFrame):
        "if dataframe return original"
        return file[Mi-1][st-vvst:ed-vvst]
    
    
"the length of the input list must be the mknumber*2, in 2d situation"
def get_error(iptlist):
    stderr = []
    for i in range(len(iptlist)):
        if(i%2 == 0):
            stdele = math.sqrt(iptlist[i]**2 + iptlist[i+1]**2)
            stderr.append(stdele)

    return stderr


def get_errordf(iptdf):
    stderrlist = []
    for idx in range(len(iptdf)):

        stderr = get_error(iptdf.loc[idx])

        stderrlist.append(stderr)

    return pd.DataFrame(stderrlist)

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
    
    return x

