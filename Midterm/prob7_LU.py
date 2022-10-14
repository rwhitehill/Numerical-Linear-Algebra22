#!/usr/bin/env python3

import numpy as np
import time

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

    n = 500
    bL = 1
    bU = 1
    A = np.zeros((n,n))
    for i in range(n):
        if i > bL-1:
            A[i,i-1] = -1
        A[i,i] = 4
        if i != n-bU:
            A[i,i+1] = -1
    A_inv = np.linalg.inv(A)

    b1 = np.ones(n)
    start = time.time()
    x1   = solve_x(A,b1)
    end = time.time()
    x1_actual = np.matmul(A_inv,b1)
    print('calculation 1 time:',end - start)
    print('x from factorization:\n',x1)
    print('x from inversion:\n',x1_actual)
    print('difference magnitude:\n',np.linalg.norm(x1_actual-x1,2))

    print()
   
    b2 = np.array([i for i in range(1,n+1)])
    start = time.time()
    x2   = solve_x(A,b2)
    end = time.time()
    x2_actual = np.matmul(A_inv,b2)
    print('calculation 2 time:',end - start)
    print('x from factorization:\n',x2)
    print('x from inversion:\n',x2_actual)
    print('difference magnitude:\n',np.linalg.norm(x2_actual-x2,2))




