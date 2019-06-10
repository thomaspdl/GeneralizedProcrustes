import numpy as np
from numpy import linalg as LA
from random import seed
import matplotlib.pyplot as plt
import pandas as pd
from math import floor, sqrt, log

from generateOneInstance import generate

from tempfile import TemporaryFile


#####################################
# dimension of the problem
#####################################
p = 3 #dimension
k = 100 #number of points



#####################################
# generate cloud
#####################################
X = np.zeros((p,k))
for l in range(1,k):
    v = np.random.randn(p)
    #v = v/LA.norm(v)
    X[:,l] = v

X = X/LA.norm(X,'fro')


#####################################
# sigma and N generation
# log scale
#####################################
kmax = 16
#set of sigmas
linsigma = np.linspace(-3.8,0.7,num = 60).tolist()
sigma = [2**i for i in linsigma]
sigmaold = sigma
sigma = np.unique(np.array(sigma)).tolist()
#set of N
linN = np.linspace(2, 12, num=80).tolist()
N = [int(2**i) for i in linN]
N = np.unique(np.array(N)).tolist()
Nplot = N[0::int(floor(len(N)/10))]
sigmaplot = sigma
#####################################


yspace = np.array([u for u in range(len(sigma))])
xspace = np.array([u for u in range(len(N))])
x_labels = xspace[::8]
y_labels = yspace[::8]
Nxlabels = N[::8]
Sylabels = linsigma[::8]



#####################################
# launch "Nreal" realizations
#####################################
Nreal = 50
MatErr = np.matrix([[0] * len(N) for i in range(len(sigma))])

for i in range(Nreal):

    print 'realization: ' + str(i)
    MatErrT, MatGramErrT = generate(X, k, p, sigma, N)
    MatErrT = np.asarray(MatErrT)

    MatErr = (float(1)/float(i+1))*np.matrix(MatErrT) + (float(i)/float(i+1))*MatErr


#####################################
#save the file
#####################################
outfile = TemporaryFile()
np.savetxt('ErrMatrix.txt', MatErr)

