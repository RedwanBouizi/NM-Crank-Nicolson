import numpy as np
import sys
from datetime import datetime  # speed of the algorithm
from math import asinh, sinh, exp
from scipy.sparse import spdiags, csc_matrix, lil_matrix
from scipy.sparse.linalg import inv
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


####################   Used for the Non Uniform Grid   ########################
#Upwind schemes
#this one will be useful for the Neumann condition at s= S_min
def gamma_s(V_ds, i, pos):
    if pos == 0:
        return (-2 * V_ds[i + 1] - V_ds[i + 2]) / (V_ds[i + 1] * (V_ds[i + 1] + V_ds[i + 2]))
    elif pos == 1:
        return (V_ds[i + 1] + V_ds[i + 2]) / (V_ds[i + 1] * V_ds[i + 2])
    elif pos == 2:
        return -V_ds[i + 1] / (V_ds[i + 2] * (V_ds[i + 1] + V_ds[i + 2]))
    else:
        print("Wrong pos")
        

#this one will be useful for the Neumann condition at s= S_max
def alpha_s(V_ds, i, pos):
    if pos == -2:
        return V_ds[i] / (V_ds[i - 1] * (V_ds[i - 1] + V_ds[i]))
    elif pos == -1:
        return (-V_ds[i - 1] - V_ds[i]) / (V_ds[i - 1] * V_ds[i])
    elif pos == 0:
        return (V_ds[i - 1] + 2 * V_ds[i]) / (V_ds[i] * (V_ds[i - 1] + V_ds[i]))
    else:
        print("Wrong pos")


#Central schemes
def beta_s(V_ds, i, pos):
    if pos == -1:
        return -V_ds[i + 1] / (V_ds[i] * (V_ds[i] + V_ds[i + 1]))
    elif pos == 0:
        return (V_ds[i + 1] - V_ds[i]) / (V_ds[i] * V_ds[i + 1])
    elif pos == 1:
        return V_ds[i] / (V_ds[i + 1] * (V_ds[i] + V_ds[i + 1]))
    else:
        print("Wrong pos")


def delta_s(V_ds, i, pos):
    if pos == -1:
        return 2 / (V_ds[i] * (V_ds[i] + V_ds[i + 1]))
    elif pos == 0:
        return -2 / (V_ds[i] * V_ds[i + 1])
    elif pos == 1:
        return 2 / (V_ds[i + 1] * (V_ds[i] + V_ds[i + 1]))
    else:
        print("Wrong pos")


def CN_price(T, K, S_0, r, sigma, S_min, S_max, N, M, bc, gridType='U'):
    start = datetime.now()

    ##### Grid generation
    dt = float(T) / N
    V_time = [i * dt for i in range(N + 1)]

    #Uniform grid
    if gridType == 'U':
        ds = float(S_max - S_min) / M
        V_S = [S_min + j * ds for j in range(M + 1)]

    # Non-Uniform grid
    elif gridType =='NU':
        alpha = 1  # -> uniform grid as alpha -> infinity
        c = asinh((S_max - K) / alpha), asinh((S_min - K) / alpha)
        V_xi = np.linspace(0, 1, M + 1, endpoint=True)
        V_S = [K + alpha * sinh(c[0] * xi + c[1] * (1. - xi)) for xi in V_xi]

    else:
        print("gridType not implemented")
        sys.exit(0)

    try:
        index = V_S.index(S_0)
    except ValueError:
        V_S.append(S_0)
        V_S.sort()
        index = V_S.index(S_0)
        M += 1

    V_S = np.array(V_S)
    V_ds = [V_S[i + 1] - V_S[i] for i in range(len(V_S) - 1)]
    V_ds = np.array(V_ds)

    V_a = [0.5 * r * dt * V_S[i] * beta_s(V_ds, i - 1, -1) + (dt / 4) * (sigma * V_S[i]) ** 2 * delta_s(V_ds, i - 1, -1) for i in range(1, M)]
    V_b = [0.5 * r * dt * V_S[i] * beta_s(V_ds, i - 1, 0) + (dt / 4) * (sigma * V_S[i]) ** 2 * delta_s(V_ds, i - 1, 0) - r * dt / 2 for i in range(1, M)]
    V_c = [0.5 * r * dt * V_S[i] * beta_s(V_ds, i - 1, 1) + (dt / 4) * (sigma * V_S[i]) ** 2 * delta_s(V_ds, i - 1, 1) for i in range(1, M)]

    # We extend the vectors according to the spdiags constructor
    V_a.append(0.)
    V_a.append(0.)

    V_b.insert(0, 0.)
    V_b.append(0.)

    V_c.insert(0, 0.)
    V_c.insert(0, 0.)

    V_a = np.array(V_a)
    V_b = np.array(V_b)
    V_c = np.array(V_c)

    ##### Set Matrices
    #representation of the PDE

    data_A = np.array([-V_a, 1 - V_b, -V_c])
    data_B = np.array([V_a, 1 + V_b, V_c])
    idx = np.array([-1, 0, 1])
    A = spdiags(data_A, idx, M+1, M+1)
    B = spdiags(data_B, idx, M+1, M+1)
    A = csc_matrix(A)
    B = csc_matrix(B)

    UU = np.zeros((N + 1, M + 1))
    UU[N, :] = [np.maximum(V_S[j] - K, 0) for j in range(0, M + 1)]

    if bc == 'D':  #Dirichlet
        A[0, 0] = 1
        A[0, 1] = 0
        A[M, M - 1] = 0
        A[M, M] = 1

        B[0, 0] = 0
        B[0, 1] = 0
        B[M, M - 1] = 0
        B[M, M] = 1

        UU[N, M] = np.maximum(S_max - K, 0)

    elif bc == 'N':
        # Neumann, we use the upwind scheme for the boundaries 
        # and central scheme for the first interior points
        A[0, 0] = 1 -0.5 * r * dt * V_S[0] * gamma_s(V_ds, 0, 0) + r * dt / 2
        A[0, 1] =   -0.5 * r * dt * V_S[0] * gamma_s(V_ds, 0, 1)
        A[0, 2] =   -0.5 * r * dt * V_S[0] * gamma_s(V_ds, 0, 2)

        A[1, 0] =   -0.5 * r * dt * V_S[1] * beta_s(V_ds, 0, -1)
        A[1, 1] = 1 -0.5 * r * dt * V_S[1] * beta_s(V_ds, 0, 0) + r * dt / 2
        A[1, 2] =   -0.5 * r * dt * V_S[1] * beta_s(V_ds, 0, 1)

        A[M - 1, M - 2] =   -0.5 * r * dt * V_S[M - 1] * beta_s(V_ds, M - 2, -1)
        A[M - 1, M - 1] = 1 -0.5 * r * dt * V_S[M - 1] * beta_s(V_ds, M - 2, 0) + r * dt / 2
        A[M - 1, M]     =   -0.5 * r * dt * V_S[M - 1] * beta_s(V_ds, M - 2, 1)

        A[M, M - 2] =   -0.5 * r * dt * V_S[M] * alpha_s(V_ds, M - 1, -2)
        A[M, M - 1] =   -0.5 * r * dt * V_S[M] * alpha_s(V_ds, M - 1, -1)
        A[M, M]     = 1 -0.5 * r * dt * V_S[M] * alpha_s(V_ds, M - 1, 0) + r * dt / 2
        
        
        B[0, 0] = 1 + 0.5 * r * dt * V_S[0] * gamma_s(V_ds, 0, 0) - r * dt / 2
        B[0, 1] =     0.5 * r * dt * V_S[0] * gamma_s(V_ds, 0, 1)
        B[0, 2] =     0.5 * r * dt * V_S[0] * gamma_s(V_ds, 0, 2)

        B[1, 0] =     0.5 * r * dt * V_S[1] * beta_s(V_ds, 0, -1)
        B[1, 1] = 1 + 0.5 * r * dt * V_S[1] * beta_s(V_ds, 0, 0) - r * dt / 2
        B[1, 2] =     0.5 * r * dt * V_S[1] * beta_s(V_ds, 0, 1)

        B[M - 1, M - 2] =     0.5 * r * dt * V_S[M - 1] * beta_s(V_ds, M - 2, -1)
        B[M - 1, M - 1] = 1 + 0.5 * r * dt * V_S[M - 1] * beta_s(V_ds, M - 2, 0) - r * dt / 2
        B[M - 1, M]     =     0.5 * r * dt * V_S[M - 1] * beta_s(V_ds, M - 2, 1)

        B[M, M - 2] =     0.5 * r * dt * V_S[M] * alpha_s(V_ds, M - 1, -2)
        B[M, M - 1] =     0.5 * r * dt * V_S[M] * alpha_s(V_ds, M - 1, -1)
        B[M, M]     = 1 + 0.5 * r * dt * V_S[M] * alpha_s(V_ds, M - 1, 0) - r * dt / 2

    else:
        print("Boundary condition not implemented")
        sys.exit(0)


    ##### Backward Solving
    inv_A = inv(A)
    for i in range(N-1, -1, -1):
        UU[i] = inv_A * B * UU[i + 1]
    V_price = UU[0]


    end = datetime.now()
    time = (end - start).total_seconds()

    return [V_S, V_time, UU, V_price, V_price[index], time]
