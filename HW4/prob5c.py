#!/usr/bin/env python3

import numpy as np

n = 10
A = np.zeros([n,n])
for i in range(n):
    if i != 0:
        A[i,i-1] = -1.0
    if i != n-1:
        A[i,i+1] = -1.0
    A[i,i] = 2.0

shift = 2.28562968
I = np.eye(n)
v = np.random.randint(-10,10,n)
v = v/np.linalg.norm(v,2)
iters = 0
while True:
    u = np.linalg.solve((A-shift*I),v)
    lam = np.dot(u,v)
    u = u/np.linalg.norm(u)

    if np.all(v - u) < 1e-6:
        break
    elif iters > 1e5:
        break
   
    v = u
    iters += 1

np.set_printoptions(precision=3)
print()
print('\teigenvector:',v)
print()
