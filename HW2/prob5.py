#!/usr/bin/env python3

import numpy as np


def LU_factorize_pivot(A):
    n = np.shape(A)[0]
    
    P = np.identity(n)
    L = np.zeros(np.shape(A)) 
    U = np.copy(A)

    for j in range(n):
        L[j,j] = 1
        switch_idx = j+np.argmax(np.abs(U[j:,j]))
        if switch_idx != j:
            temp = np.copy(U[switch_idx,:])
            U[switch_idx,:] = U[j,:]
            U[j,:] = temp

            temp = np.copy(P[switch_idx,:])
            P[switch_idx,:] = P[j,:]
            P[j,:] = temp

            temp = np.copy(L[j,:j])
            L[j,:j] = L[switch_idx,:j] 
            L[switch_idx,:j] = temp
        for i in range(j+1,n):
            L[i,j] = U[i,j]/U[j,j]
        for l in range(j+1,n):
            for m in range(j,n):
                U[l,m] = U[l,m] - U[j,m]*L[l,j]
        
    return P,L,U


if __name__ == '__main__':

    A = np.array([
        [4.0, 7.0,3.0],
        [1.0,3.0,2.0],
        [2.0,-4.0,-1.0],
        ])

    P,L,U = LU_factorize_pivot(A)

    np.set_printoptions(precision=3)
    print('\nP=\n{}\n'.format(P))
    print('\nL=\n{}\n'.format(L))
    print('U=\n{}\n'.format(U))

