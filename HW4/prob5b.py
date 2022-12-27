#!/usr/bin/env

import numpy as np

n = 10 
A = np.zeros([n,n])
for i in range(n):
    if i != 0:
        A[i,i-1] = -1.0
    if i != n-1:
        A[i,i+1] = -1.0
    A[i,i] = 2.0

A_old = np.copy(A)
iters = 0
while True:
    Q,R = np.linalg.qr(A_old,mode='complete')
    A1 = R@Q

    if np.all(np.abs(A1 - A_old) < 1e-6):
        break
    elif iters > 1e6:
        break

    A_old = A1
    iters += 1

np.set_printoptions(precision=3)
print()
print('\teigenvalues:',np.array([A1[i,i] for i in range(n)]))
print()
