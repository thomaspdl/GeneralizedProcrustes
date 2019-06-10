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
Ns = [2**u for u in range(1,15)]

###########################
#save the file

Ns = np.loadtxt('Ns.txt')
Sigma2Mean = np.loadtxt('Sigma2Mean.txt')
Sigma2Var = np.loadtxt('Sigma2Var.txt')
Sigma2VarPred = np.loadtxt('Sigma2VarPred.txt')

ErrMean = np.subtract(np.ones(Ns.size), np.divide(Sigma2Mean,sigma**2))
ErrVar  = np.subtract(np.divide(Sigma2VarPred,sigma**2),np.divide(Sigma2Var,sigma**2))



def myticks(x,pos):

    if x == 0: return "$0$"

    exponent = int(np.log10(x)) - 1
    coeff = x/10**exponent

    return r"${:2.0f} \times 10^{{ {:2d} }}$".format(coeff,exponent)



x = np.arange(0, 1, .01)

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
plt.rcParams["font.family"] = "Times New Roman"

fig,ax = plt.subplots(figsize=(22, 14))
ax.loglog(Ns,ErrMean,'ro',markersize=16)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel('Number of observations N', fontsize=36)
plt.ylabel('Relative Error',fontsize=36)
plt.title(' Relative Error of the estimator of $\sigma^2$', fontsize=40)
plt.savefig('/Users/thomaspumir/Dropbox/PumirBoumal/GeneralizedProcrustesProblem/writeup/ArXiv/BiasEstimatorSigma.png')
plt.show()



# ##################################
# ##################################
#
#
#
# fig, ax = plt.subplots(figsize = (22, 14))
#
# plt.xticks(x_labels,[str(u) for u in Nxlabels],fontsize=25)
# plt.yticks(y_labels,['$2^{' + str(round(u, 1)) + '}$' for u in Sylabels],fontsize=25)
#
# plt.title('Estimation Error ',fontsize=40)
# plt.xlabel('Number of observations $N$',fontsize=36)
# plt.ylabel('Noise level $\sigma$',fontsize=36)
# plt.legend(loc='lower right',prop={'size': 30},frameon=False)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes('right', size='5%', pad=0.05)
# cbar = fig.colorbar(im, cax=cax, orientation='vertical')
# cbar.ax.tick_params(labelsize=25)
#
#
#
# ##################################
# ##################################








# fig,ax = plt.subplots(figsize=(22, 14))
# #plt.subplot(121)
# ax.plot(Ns,np.divide(Sigma2Var,sigma**2),'ro',label="Empirical variance",markersize=16)
# ax.plot(Ns,np.divide(Sigma2VarPred,sigma**2),'b',label="Predicted variance")
# #ax.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))
# #plt.yticks(Sigma2Var,[str(round(u*(10**6), 1)) for u in Sigma2Var],fontsize=12)
# #plt.ylim(0,1.25*max(Sigma2Var))
# ax.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))
# plt.xticks(fontsize=25)
# plt.yticks(fontsize=25)
# plt.xlabel('Number of observations N', fontsize=40)
# plt.ylabel('Relative variance', fontsize=40)
# plt.title('Variance of the estimator', fontsize=48)
# plt.legend(prop={'size': 30})
# # plt.subplot(122)
# # plt.plot(Ns,ErrVar,'r',label="Variance")
# # #plt.plot(Ns,np.divide(np.multiply(Sigma2VarPred,Ns),sigma**2),'b',label="Predicted Variance")
# # #plt.ylim(100,160)
# # plt.xlabel('Number of Samples')
# # plt.ylabel('Relative Variance Estimation Error')
# # plt.legend()
# plt.savefig('/Users/thomaspumir/Dropbox/PumirBoumal/GeneralizedProcrustesProblem/writeup/ArXiv/VarianceEstimatorSigma.png')
# plt.show()


# plt.figure(figsize=(5,5))
# plt.plot(Ns,np.subtract(np.divide(Sigma2VarPred,sigma**2),np.divide(Sigma2Var,sigma**2)))
# plt.show()
