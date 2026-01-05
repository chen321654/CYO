import numpy as np
from cyo.utils.chaos import get_p_and_q
from PIL import Image

def encryption(image, key):
    """
    encrypt the image
    
    :param image: 说明
    :param key: 密钥
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    I1 = image[:, :, 0].copy()
    I2 = image[:, :, 1].copy()
    I3 = image[:, :, 2].copy()

    M, W, _ = image.shape

    Sum = M * W
    p1, q1 = key.split(",")
    p1, q1 = np.float32(p1), np.float32(q1)
    # p1 = 0.5479
    # q1 = 0.4014
    p, q = get_p_and_q(Sum, p1, q1)

    S1 = np.floor(np.multiply(256, np.divide(np.add(p, q), 2)))
    S2 = np.floor(np.multiply(256, p))

    q1 = q[:Sum]
    q2 = q[Sum: 2*Sum]
    q3 = q[2*Sum: len(p)]

    RR = np.reshape(q1, (W, M), order='F').T
    RG = np.reshape(q2, (W, M), order='F').T
    RB = np.reshape(q3, (W, M), order='F').T
    t = np.gcd(M, W)

    TL = np.zeros(M * 3, dtype=image.dtype)
    TH = np.zeros(W * 3, dtype=image.dtype)
    Min = min(M, W)
    for i in range(0, M, t):
        for j in range(0, W, t):
            wi = int(np.floor(RR[i, j] * M)) + 1
            wj  = int(np.floor(RG[i, j] * W)) + 1
            wy = int(np.floor(RB[i, j] * Min)) + 1

            TH[:wy] = I3[wi, W-wy: W]
            TH[wy: W+wy] = I1[wi, :]
            TH[W+wy: 2*W+wy] = I2[wi, :]
            TH[2*W+wy: W*3] = I3[wi, :W-wy]
            I1[wi, :] = TH[:W]
            I2[wi, :] = TH[W: W*2]
            I3[wi, :] = TH[W*2: W*3]

            TL[:wy] = I3[M-wy:M, wj]
            TL[wy: M+wy] = I1[:, wj]
            TL[M+wy: 2*M+wy] = I2[:, wj]
            TL[2*M+wy: M*3] = I3[:M-wy, wj]
            I1[:, wj] = TL[:M]
            I2[:, wj] = TL[M: M*2]
            I3[:, wj] = TL[2*M: M*3]
    
    flat_I1 = I1.flatten(order='F')
    flat_I2 = I2.flatten(order='F')
    flat_I3 = I3.flatten(order='F')

    A = np.vstack((flat_I1, flat_I2, flat_I3)).flatten(order='F')

    total_len = len(A)
    B = np.zeros(total_len)
    C = np.zeros(total_len)

    B[-1] = (A[-1] + S1[-1]) % 256

    for i in range(total_len - 2, -1, -1):
        B[i] = (B[i+1] + S1[i] + A[i]) % 256

    C[0] = (B[0] + S2[0]) % 256

    for i in range(1, total_len):
        C[i] = (C[i-1] + B[i] + S2[i]) % 256

    MW = M * W

    I1_new = C[0:MW].reshape((M, W), order='F').astype(np.uint8)
    I2_new = C[MW:2*MW].reshape((M, W), order='F').astype(np.uint8)
    I3_new = C[2*MW:].reshape((M, W), order='F').astype(np.uint8)

    image[:, :, 0] = I1_new
    image[:, :, 1] = I2_new
    image[:, :, 2] = I3_new

    return image

