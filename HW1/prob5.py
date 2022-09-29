#!/usr/bin/env python3

import numpy as np


###---Part (a)---###
print('\nPart (a) \n')

x = np.array([3,-4,0,3/2])
print('1-norm:   %.2f'%np.linalg.norm(x,ord=1))
print('2-norm:   %.2f'%np.linalg.norm(x,ord=2))
print('inf-norm: %.2f'%np.linalg.norm(x,ord=np.inf))

###---Part (b)---###
print('\n\nPart (b) \n')

A = np.array([[2,-1,0],[-1,2,-1],[0,-1,2]])
print('1-norm:   %.2f'%np.linalg.norm(A,ord=1))
print('2-norm:   %.2f'%np.linalg.norm(A,ord=2))
print('inf-norm: %.2f'%np.linalg.norm(A,ord=np.inf))

print()