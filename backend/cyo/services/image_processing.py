import numpy as np
from cyo.utils.chaos import get_R_matrix

def encryption(image, key):
    """
    encrypt the image
    
    :param image: the image will be encrypted, format: numpy
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

    RR, RG, RB, p, q = get_R_matrix(W, M, p1, q1)
    t = np.gcd(M, W)

    TL = np.zeros(M * 3, dtype=image.dtype)
    TH = np.zeros(W * 3, dtype=image.dtype)
    Min = min(M, W)
    for i in range(0, M, t):
        for j in range(0, W, t):
            wi = int(np.floor(RR[i, j] * M)) + 1
            wj  = int(np.floor(RG[i, j] * W)) + 1
            wy = int(np.floor(RB[i, j] * Min)) + 1

            TH[:wy] = I3[wi, W-wy: W].copy()
            TH[wy: W+wy] = I1[wi, :].copy()
            TH[W+wy: 2*W+wy] = I2[wi, :].copy()
            TH[2*W+wy: W*3] = I3[wi, :W-wy].copy()
            I1[wi, :] = TH[:W].copy()
            I2[wi, :] = TH[W: W*2].copy()
            I3[wi, :] = TH[W*2: W*3].copy()

            TL[:wy] = I3[M-wy:M, wj].copy()
            TL[wy: M+wy] = I1[:, wj].copy()
            TL[M+wy: 2*M+wy] = I2[:, wj].copy()
            TL[2*M+wy: M*3] = I3[:M-wy, wj].copy()
            I1[:, wj] = TL[:M].copy()
            I2[:, wj] = TL[M: M*2].copy()
            I3[:, wj] = TL[2*M: M*3].copy()
    
    flat_I1 = I1.flatten(order='F')
    flat_I2 = I2.flatten(order='F')
    flat_I3 = I3.flatten(order='F')

    A = np.vstack((flat_I1, flat_I2, flat_I3)).flatten(order='F')

    total_len = len(A)
    B = np.zeros(total_len)
    C = np.zeros(total_len)
    S1 = np.floor(np.multiply(256, np.divide(np.add(p, q), 2)))
    S2 = np.floor(np.multiply(256, p))

    B[-1] = (A[-1] + S1[-1]) % 256
    for i in range(total_len - 2, -1, -1):
        B[i] = (B[i+1] + S1[i] + A[i]) % 256

    C[0] = (B[0] + S2[0]) % 256
    for i in range(1, total_len):
        C[i] = (C[i-1] + B[i] + S2[i]) % 256

    I1_new = C[0:Sum].reshape((M, W), order='F').astype(np.uint8)
    I2_new = C[Sum:2*Sum].reshape((M, W), order='F').astype(np.uint8)
    I3_new = C[2*Sum:].reshape((M, W), order='F').astype(np.uint8)

    image[:, :, 0] = I1_new
    image[:, :, 1] = I2_new
    image[:, :, 2] = I3_new

    return image


def decryption(image, key):
    """
    decrypt the image with key
    
    :param image: encrypted image, format: numpy
    :param key: chaos key
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    I1 = image[:, :, 0].copy()
    I2 = image[:, :, 1].copy()
    I3 = image[:, :, 2].copy()

    M, N, _ = image.shape
    Sum = M * N
    p1, q1 = key.split(",")
    p1, q1 = np.float32(p1), np.float32(q1)

    RR, RG, RB, p, q = get_R_matrix(N, M, p1, q1)
    t = np.gcd(M, N)

    C = np.concatenate([I1.flatten(order='F'), 
                        I2.flatten(order='F'), 
                        I3.flatten(order='F')])
    
    D = np.zeros_like(C)
    E = np.zeros_like(C)
    total_pixels = 3 * M * N
    S1 = np.floor(np.multiply(256, np.divide(np.add(p, q), 2)))
    S2 = np.floor(np.multiply(256, p))
    # --- D Loop (MATLAB: 3*M*N:-1:2) ---
    D[1:] = (C[1:] - C[:-1] - S2[1:]) % 256
    # 处理边界 D(1)
    D[0] = (C[0] - S2[0]) % 256
    
    # --- E Loop (MATLAB: 1:3*M*N-1) ---
    E[:-1] = (D[:-1] - D[1:] - S1[:-1]) % 256
    # 处理边界 E(end)
    E[-1] = (D[-1] - S1[-1]) % 256
    
    
    I1_flat = E[0::3] # 取出所有 R
    I2_flat = E[1::3] # 取出所有 G
    I3_flat = E[2::3] # 取出所有 B
    
    I1 = I1_flat.reshape((M, N), order='F')
    I2 = I2_flat.reshape((M, N), order='F')
    I3 = I3_flat.reshape((M, N), order='F')

    t = np.gcd(M, N)
    Min = min(M, N)
    
    for i in range(M - t, -1, -t):
        for j in range(N - t, -1, -t):
            wi = int(np.floor(RR[i, j] * M)) + 1
            wj = int(np.floor(RG[i, j] * N)) + 1
            wy = int(np.floor(RB[i, j] * Min)) + 1
            
            TL = np.concatenate([I1[:, wj], I2[:, wj], I3[:, wj]])
            I1[:, wj] = TL[wy : M + wy].copy()
            I2[:, wj] = TL[M + wy : 2*M + wy].copy()
            part1 = TL[2*M + wy :].copy()
            part2 = TL[0 : wy].copy()
            I3[:, wj] = np.concatenate([part1, part2])
            
            TH = np.concatenate([I1[wi, :], I2[wi, :], I3[wi, :]])
            I1[wi, :] = TH[wy : N + wy].copy()
            I2[wi, :] = TH[N + wy : 2*N + wy].copy()
            part1_row = TH[2*N + wy :].copy()
            part2_row = TH[0 : wy].copy()
            I3[wi, :] = np.concatenate([part1_row, part2_row])

    image_de = np.stack([I1, I2, I3], axis=2).astype(np.uint8)
    return image_de