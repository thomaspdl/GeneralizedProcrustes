import numpy as np
from numpy import linalg as LA
from random import seed
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
import pandas as pd
from math import floor, sqrt, log

from tempfile import TemporaryFile


d = 3
k = 250
sigma = 1
Ns = [5000,6000,7000,8000,9000,10000]

###########################
#save the file

Ns = np.loadtxt('Ns250.txt')
Sigma2Mean = np.loadtxt('Sigma2Mean250.txt')
Sigma2Var = np.loadtxt('Sigma2Var250.txt')
Sigma2VarPred = np.loadtxt('Sigma2VarPred250.txt')

ErrMean = np.subtract(np.ones(Ns.size), np.divide(Sigma2Mean,sigma**2))
ErrVar  = np.subtract(np.divide(Sigma2VarPred,sigma**2),np.divide(Sigma2Var,sigma**2))



def myticks(x,pos):

    if x == 0: return "$0$"

    exponent = int(np.log10(x)) - 1
    coeff = x/10**exponent

    return r"${:2.0f} \times 10^{{ {:2d} }}$".format(coeff,exponent)



x = np.arange(0, 1, .01)

fig,ax = plt.subplots(figsize=(22, 14))
ax.plot(Ns,ErrMean,'ro',markersize=16)
#ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
#plt.ylim(0,1.5*max(ErrMean))
#plt.yticks()
#ax.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))
#ax.xaxis.set_major_formatter(ticker.FuncFormatter(myticks))
plt.xticks(fontsize=40)
plt.yticks(fontsize=30)
plt.ylabel('$\sigma^2 - \hat{\sigma}_{N}^{2}$', fontsize=44)
plt.xlabel('Number of Observations N', fontsize=48)
plt.title(' Bias of the estimator', fontsize=56)
plt.savefig('/Users/thomaspumir/Dropbox/PumirBoumal/GeneralizedProcrustesProblem/writeup/FinalPaper/BiasEstimatorSigma.png')
plt.show()



fig,ax = plt.subplots(figsize=(22, 14))
#plt.subplot(121)
ax.plot(Ns,np.divide(Sigma2Var,sigma**2),'ro',label="Empirical Variance",markersize=16)
ax.plot(Ns,np.divide(Sigma2VarPred,sigma**2),'b',label="Predicted Variance")
#ax.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))
#plt.yticks(Sigma2Var,[str(round(u*(10**6), 1)) for u in Sigma2Var],fontsize=12)
#plt.ylim(0,1.25*max(Sigma2Var))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))
plt.xticks(fontsize=40)
plt.yticks(fontsize=30)
plt.xlabel('Number of Observations N', fontsize=48)
plt.ylabel('Relative Variance', fontsize=44)
plt.title('Variance Of the Estimator', fontsize=56)
plt.legend(prop={'size': 36})
# plt.subplot(122)
# plt.plot(Ns,ErrVar,'r',label="Variance")
# #plt.plot(Ns,np.divide(np.multiply(Sigma2VarPred,Ns),sigma**2),'b',label="Predicted Variance")
# #plt.ylim(100,160)
# plt.xlabel('Number of Samples')
# plt.ylabel('Relative Variance Estimation Error')
# plt.legend()
plt.savefig('/Users/thomaspumir/Dropbox/PumirBoumal/GeneralizedProcrustesProblem/writeup/FinalPaper/VarianceEstimatorSigma.png')
plt.show()


# plt.figure(figsize=(5,5))
# plt.plot(Ns,np.subtract(np.divide(Sigma2VarPred,sigma**2),np.divide(Sigma2Var,sigma**2)))
# plt.show()
