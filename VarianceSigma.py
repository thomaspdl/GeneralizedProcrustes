import numpy as np
from numpy import linalg as LA
from random import seed
import matplotlib.pyplot as plt
import pandas as pd
from math import floor, sqrt, log

from tempfile import TemporaryFile


#dimensions of the problem
d = 3 #dimension
k = 250 #number of points



#generate k points of dimension d
X = np.zeros((d,k))
for l in range(1,k):
    v = np.random.randn(d)
    v = v/LA.norm(v)
    X[:,l] = v

X = X/LA.norm(X,'fro')

sigma = 1
Ns = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]

Sigma2Mean = []
Sigma2Var = []
Sigma2VarPred = []

for N in Ns:
    print '*******************************'
    print N
    print '*******************************'
    Sigma2Est = []
    for nreal in range(1000):
        if nreal % 100 == 0:
            print nreal

        M = np.zeros((k,k))
        for l in range(N):

            # #generate rotation
            # A = np.random.randn(d, d)
            # Q, R = np.linalg.qr(A)
            # sign_diag = np.sign(np.diag(np.diag(R)))
            # Q = np.dot(Q, sign_diag)

            #generate noise
            E = sigma * np.random.randn(d, k)

            #generate observation
            Y = X + E

            #average Gram
            M = M + (1./N)*np.dot(np.transpose(Y),Y)


        #compute d largest eigenvalues
        w, v = LA.eig(M)
        w = w.real
        w[::-1].sort()
        #estimate sigma2
        sigma2 = 1./(d*(k - d))*(np.trace(M) - np.sum(w[:d]))
        Sigma2Est.append(sigma2)


    #print w[:d]
    Sigma2Mean.append(np.mean(Sigma2Est))
    Sigma2Var.append(np.var(Sigma2Est))
    Sigma2VarPred.append(float(2*k/float(N*d*(k - d)**2)*sigma**4))
    #estimate sigma



print Sigma2Mean
print Sigma2Var
print Sigma2VarPred

###########################
#save the file
outfile = TemporaryFile()
np.savetxt('Ns.txt', Ns)
np.savetxt('Sigma2Mean.txt', Sigma2Mean)
np.savetxt('Sigma2Var.txt', Sigma2Var)
np.savetxt('Sigma2VarPred.txt', Sigma2VarPred)



plt.figure(figsize=(5,5))
plt.subplot(121)
plt.plot(Ns,Sigma2Mean,'r')
plt.subplot(122)
plt.plot(Ns,Sigma2Var,'r')
plt.plot(Ns,Sigma2VarPred,'b')
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.05)
#plt.colorbar(cax=cax)
plt.show()
