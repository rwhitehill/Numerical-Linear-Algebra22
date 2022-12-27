#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def my_spy(ax,A,marker='o'):
    n,m = np.shape(A)
    x = []
    y = []
    for i in range(n):
        for j in range(m):
            if A[j,i] < 1.0e-10:
                continue
            else:
                x.append(i)
                y.append(j)

    ax.scatter(x,y,color='k',marker='o',facecolor='none',s=10)
    ax.set_ylim(n,-1)
    ax.set_xlim(-1,n)


nrows=1;ncols=2
fig,ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(7*ncols,5*nrows))

n = 100
A = np.random.randint(-5,5,(n,n))

my_spy(ax[0],A)

N = 1000
for i in range(N):
    Q,R = np.linalg.qr(A,mode='complete')
    A = R@Q

my_spy(ax[1],A)

fig.tight_layout()
fig.savefig('prob4.pdf',bbox_inches='tight')
