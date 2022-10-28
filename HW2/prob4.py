#!/usr/bin/env python3

import numpy as np

def LU_factorize(A):
    n = np.shape(A)[0]

    L = np.zeros(np.shape(A)) 
    U = np.copy(A)

    for j in range(n):
        L[j,j] = 1
        for i in range(j+1,n):
            L[i,j] = U[i,j]/U[j,j]
        for l in range(j+1,n):
            for m in range(j,n):
                U[l,m] = U[l,m] - U[j,m]*L[l,j]
    return L,U

def solve_x(A,b): 
    n = np.shape(A)[0]

    L,U = LU_factorize(A)
   
    y = np.zeros(n)  
    y[0] = b[0]/L[0,0]
    for i in range(1,n):
        temp = np.array([y[k]*L[i,k] for k in range(i)])
        y[i] = (b[i] - np.sum(temp))/L[i,i]

    x = np.zeros(n)
    x[-1] = y[-1]/U[-1,-1]
    for i in range(n-2,-1,-1):
        temp = np.array([x[k]*U[i,k] for k in range(i+1,n)])
        x[i] = (y[i] - np.sum(temp))/U[i,i]


    return x

if __name__ == '__main__':

    A = np.array([
        [ 4.0, 1.0,-1.0,1.0],
        [ 1.0, 4.0,-1.0,-1.0],
        [-1.0,-1.0, 5.0,1.0],
        [ 1.0,-1.0, 1.0,3.0] 
        ],)
    b = np.array([-2,-1,0,1]) 
    L,U = LU_factorize(A)
    x = solve_x(A,b)
    
    np.set_printoptions(precision=3)
    print('\nL=\n{}\n'.format(L))
    print('U=\n{}\n'.format(U))
    print('x={}\n'.format(x))

