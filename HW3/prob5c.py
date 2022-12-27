#!usr/bin/env python3

import numpy as np

def least_squares_SVD(A,b):
    U,S,VT = np.linalg.svd(A,full_matrices=False)

    y = U.T @ b
    w = np.array([y[i]/S[i] for i in range(len(y))])
    x = VT.T @ w

    return x 

if __name__ == '__main__':

    A = np.array([
        [2.0,1.0],
        [1.0,1.0],
        [2.0,1.0]
        ])
    b = np.array([12.0,6.0,18.0])

    x = least_squares_SVD(A,b)

    np.set_printoptions(precision=3)
    print('\n  x = {}'.format(x))
    print('  residual: {:.3f}\n'.format(np.linalg.norm(A@x-b)))



