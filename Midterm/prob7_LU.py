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
   
    print('L\n',L)
    print('U\n',U) 
    print(np.matmul(L,U)-A)

    y = np.zeros(n)  
    y[0] = b[0]/L[0,0]
    for i in range(1,n):
        temp = np.array([y[k]*L[i,k] for k in range(i)])
        y[i] = (b[i] - np.sum(temp))/L[i,i]

    print(np.matmul(L,y)-b)
    
    x = np.zeros(n)
    x[-1] = y[-1]/U[-1,-1]
    for i in range(n-2,-1,-1):
        temp = np.array([x[k]*U[i,k] for k in range(i+1,n)])
        x[i] = (y[i] - np.sum(temp))/U[i,i]

    print(np.matmul(U,x)-y)

    return x

if __name__ == '__main__':
    
    n = 5
    A = np.zeros((n,n))
    for i in range(n):
        if i != 0:
            A[i,i-1] = -1
        A[i,i] = 4
#        if i < n-2:
#            A[i,i+2] = -1
        if i > 1:
            A[i,i-2] = -1
        if i != n-1:
            A[i,i+1] = -1

    print(A)
    
    b = np.ones(n)
#    b = np.array([i for i in range(1,n+1)])

    start = time.time()
    L,U = LU_factorize(A)
    x   = solve_x(A,b)
    end = time.time()
    
    print(end - start)





