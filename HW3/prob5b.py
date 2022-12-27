#!usr/bin/env python3

import numpy as np

def QR_modifiedGS(A):
    m,n = np.shape(A)
    R = np.zeros([n,n])
    Q = np.copy(A)
    for i in range(n):
        v = Q[:,i]
        for j in range(i):
            R[j,i] = v.T @Q[:,j]
            Q[:,i] -= R[j,i]*Q[:,j]
        R[i,i] = np.linalg.norm(Q[:,i])
        Q[:,i] = v/R[i,i]

    return Q,R

def least_squares_QR(A,b): 
    m,n = np.shape(A)
    Q,R = QR_modifiedGS(A)
    
    y = Q.T @ b

    x = np.zeros(n)
    x[-1] = y[-1]/R[-1,-1]
    for i in range(n-2,-1,-1):
        temp = np.array([x[k]*R[i,k] for k in range(i+1,n)])
        x[i] = (y[i] - np.sum(temp))/R[i,i]

    return x


if __name__ == '__main__':

    A = np.array([
        [2.0,1.0],
        [1.0,1.0],
        [2.0,1.0]
        ])
    b = np.array([12.0,6.0,18.0])
   
    x = least_squares_QR(A,b)

    np.set_printoptions(precision=3)
    print('\n  x = {}'.format(x))
    print('  residual: {:.3f}\n'.format(np.linalg.norm(A@x-b)))

