import numpy as np
from numpy import sin as sin
from numpy import pi as pi

def get_p_and_q(Sum: int, p1: float, q1: float):
    p = np.zeros(Sum*3 + 1000)
    q = np.zeros(Sum*3 + 1000)
    a = 19.3
    k = 9.3
    p[0] = p1
    q[0] = q1
    for i in range(Sum*3 + 999):
        p[i + 1] = sin( 21 / (a*k*p[i] * (q[i] + 3) * (1 - k*p[i])))
        q[i + 1] = sin( 21 / (a*q[i] * (k*p[i + 1] + 3) * (1 - q[i])))
    
    # print(p)
    p = p[1000:Sum*3 + 1000]
    p = np.divide(np.add(p, a), 2 * a)
    p = np.divide(np.floor(p * np.pow(10, 4)), np.pow(10, 4))

    q = q[1000:Sum*3 + 1000]
    q = np.divide(q, k)
    q = np.divide(np.floor(q*np.pow(10, 4)), np.pow(10, 4)) 
    q = np.abs(q)
    return p, q


def getx_y(dot, Max, x1, y1):
    size = dot + Max * 3 + 5
    x = np.zeros(size)
    y = np.zeros(size)
    x[0] = x1
    y[0] = y1
    a = 0.6
    k = 0.8
    for n in range(dot + Max * 3):
        x[n + 1] = np.sin(21 * 1.0 / (a * (y[n] + 3) * k * x[n] * (1 - k * x[n])))
        y[n + 1] = np.sin(21 * 1.0 / (a * (k * x[n + 1] + 3) * y[n] * (1 - y[n])))
    x[:dot] = (x[:dot] + 1) / 2
    y[:dot] = (y[:dot] + 1) / 2
    return x, y  # 返回完整数组，不截断

def getk1_k2(x, y, dot, left, right, top, bottom, mark):
    H = bottom - top
    W = right - left
    Max_1 = max(right, left)
    Max_2 = max(top, bottom)
    Max_3 = max(Max_1, Max_2)
    Max = max(H, W)
    Min = min(H, W)
    k1 = np.zeros(Max_3, dtype=np.int32)
    k2 = np.zeros(Max_3, dtype=np.int32)
    for n in range(Max_3):
        k1_value = x[dot - (Max_3 - n)] * 1000
        k2_value = y[dot - (Max_3 - n)] * 1000
        k1[n] = int(np.floor(k1_value)) % Max
        k2[n] = int(np.floor(k2_value)) % Min
    if mark:
        for n in range(Max_3):
            k2[n] = int(np.floor(y[dot - (Max_3 - n)] * 1000)) % Max
    return k1, k2