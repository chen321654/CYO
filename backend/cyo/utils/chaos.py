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

def get_R_matrix(W: int, M: int, p1: float, q1: float):
    Sum = W * M
    p, q = get_p_and_q(Sum, p1, q1)

    q1 = q[:Sum]
    q2 = q[Sum: 2*Sum]
    q3 = q[2*Sum: len(p)]

    RR = np.reshape(q1, (W, M), order='F').T
    RG = np.reshape(q2, (W, M), order='F').T
    RB = np.reshape(q3, (W, M), order='F').T

    return RR, RG, RB, p, q