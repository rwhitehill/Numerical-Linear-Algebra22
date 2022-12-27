#!usr/bin/env python3

import numpy as np

A = np.array([
     [3.0,4.0],
     [0.0,1.0],
     [4.0,0.0]
     ])

m,n = np.shape(A)

i  = 0
Hs = []
for j in range(n):
    a = A[i:,j]

    u = a + np.linalg.norm(a)*np.sign(a[0])*np.array([1 if k==0 else 0 for k in range(m-i)])
    print(u)
    H = np.eye(m-i) - 2*np.outer(u,u)/np.linalg.norm(u)**2
    if i != 0:
        H = np.block([[np.eye(i), np.zeros((1,m-i))],[np.zeros((m-i,1)),H]])

    A = np.matmul(H,A)
    print(H)
    print(A)

    print()

    Hs.append(H)

    i += 1

    
