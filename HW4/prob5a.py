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

v = np.random.randint(-10,10,n)
v = v/np.linalg.norm(v,2)
lam_old = None
iters = 0
while True:
    u = A@v
    lam = np.dot(u,v)
    u = u/np.linalg.norm(u)

    if lam_old == None:
       lam_old = lam
    elif np.abs((lam_old - lam)/lam_old) < 1e-6 and np.all(v - u) < 1e-6:
        break
    elif iters > 1e5:
        break
   
    lam_old = lam
    v = u
    iters += 1

np.set_printoptions(precision=3)
print()
print('\tlamda = {:.3f}'.format(lam))
print('\teigenvector:',v)
print()
