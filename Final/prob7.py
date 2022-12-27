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

def LU_solve(A,b): 
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

def Jacobi(A,b,x0=None):
    n = np.shape(A)[0]
    if x0 is None:
        x0 = np.zeros(n)

    it = 0
    while True:
        x = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if j != i:
                    x[i] -= A[i,j]*x0[j]
            x[i] += b[i]
            x[i] /= A[i,i]
        if np.linalg.norm(A@x - b,2) < 1e-5 or it > 1e6:
            break
        else:
            x0 = np.copy(x)
            it += 1
    return x,it

def Gauss_Seidel(A,b,x0=None):
    n = np.shape(A)[0]
    if x0 is None:
        x0 = np.zeros(n)

    it = 0
    while True:
        x = np.zeros(n) 
        for i in range(n):
            for j in range(n):
                if j < i:
                    x[i] -= A[i,j]*x[j]
                elif j > i:
                    x[i] -= A[i,j]*x0[j]
            x[i] += b[i]
            x[i] /= A[i,i]
        if np.linalg.norm(A@x - b,2) < 1e-5 or it > 1e6:
            break
        else:
            x0 = np.copy(x)
            it += 1
    return x,it

def conj_gradient(A,b,x=None):
    n = np.shape(A)[0]
    if x is None:
        x = np.zeros(n)
        x[0] = 1.0
  
    r = b - A@x
    r_old = np.dot(r,r)
    p = r
    it = 0
    while True:
        if np.linalg.norm(r,2) < 1e-10 or it > 1e6:
            break
        
        coeff = np.dot(r,r)/np.dot(p,A@p)
        x += coeff*p

        r -= coeff*A@p
        r_new = np.dot(r,r)
        p = r + r_new/r_old*p

        r_old = r_new
        it += 1
    return x,it

def print_comp(file,method,x_approx,x_exact,time,it=None):
    file.write('  '+method+'\n')
    if it is not None: 
        file.write('    approximate solution ({} iterations): {}\n'.format(it,x_approx))
    else:
        file.write('    approximate solution: {}\n'.format(it,x_approx))
    file.write('    error: {:.5e}\n'.format(np.linalg.norm(x_approx - x_exact,2)))
    file.write('    time: {:.4f} s\n\n'.format(time))

if __name__ == '__main__':

    n = 500
    A = np.zeros([n,n])
    for i in range(n):
        if i > 0:
            A[i,i-1] = -1
        if i < n-1:
            A[i,i+1] = -1
        A[i,i] = 4
    b = np.array([i+1 for i in range(n)])

    t0 = time.time()
    x_true = np.linalg.inv(A)@b
    t1 = time.time()
    
    np.set_printoptions(precision=4,threshold=5)
    file = open('prob7.txt','w')

    file.write('\n')
    file.write('  Explicit inverse:\n')
    file.write('    exact solution: {}\n'.format(x_true))
    file.write('    time: {:.4f} s\n\n'.format(t1-t0))

    t0 = time.time()
    x = LU_solve(A,b)
    t1 = time.time()
    print_comp(file,'LU factorization',x,x_true,t1-t0)

    t0 = time.time()
    x,it = Jacobi(A,b)
    t1 = time.time()
    print_comp(file,'Jacobi iteration',x,x_true,t1-t0,it)

    t0 = time.time()
    x,it = Gauss_Seidel(A,b)
    t1 = time.time()
    print_comp(file,'Gauss-Seidel iteration',x,x_true,t1-t0,it)

    t0 = time.time()
    x,it = conj_gradient(A,b)
    t1 = time.time()
    print_comp(file,'Conjugate gradient iteration',x,x_true,t1-t0,it)





