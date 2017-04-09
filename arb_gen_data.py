import numpy as np
import tables
import matplotlib.pyplot as plt
import random as rd
import os

nsamp = 1000000
fname  = 'test.i16'

fid = open(fname, mode='wb')

atimes = np.linspace(1, 30000, nsamp)
atimes = atimes.astype(np.int16)
atimes.tofile(fname, "")

#plt.plot(atimes)
#plt.show()

# see if its right

fid = open(fname, mode = 'rb')
atimes2 = fid.read(nsamp*2)
#array_length = int(len(data) / NCS_RECSIZE)
atimes2 = np.ndarray(nsamp, 'i2', atimes2)


#atimes2 = np.fromfile(file=fname, dtype=np.int16)

print(atimes.shape)
print(atimes2.shape)
plt.plot(atimes2)
plt.show()
