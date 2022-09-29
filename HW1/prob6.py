#!/usr/bin/env python3

import numpy as np

###---Part (a)---###
print('\nPart (a) \n')

A = np.array([[3,0],[0,-2]])
U,Sig,Vt = np.linalg.svd(A,full_matrices=True)
print('U=\n{}\n'.format(U))
print('Sigma=\n{}\n'.format(Sig))
print('V*=\n{}\n'.format(Vt))

###---Part (b)---###
print('\n\nPart (b) \n')

A = np.array([[0,2],[0,0],[0,0],[0,0]])
U,Sig,Vt = np.linalg.svd(A,full_matrices=True)
print('U=\n{}\n'.format(U))
print('Sigma=\n{}\n'.format(Sig))
print('V*=\n{}\n'.format(Vt))

print()

