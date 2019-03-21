#####################################
# libraries
#####################################
import numpy as np
from numpy import linalg as LA
from random import seed
import matplotlib.pyplot as plt
from math import floor, sqrt, log
from sklearn import linear_model


#####################################
# dimension of the problem
#####################################
p = 3 #dimension
k = 250 #number of points


#####################################
# hyperparameters
#####################################
Thresh = 0.95 #recovery threshold
OrdinatesOrigin = 6 #Ordinate at origin for Gram recovery
alpha = 3.5/5. #portion of the graph where oracle recovery is displayed

#####################################
# generate cloud
#####################################
X = np.zeros((p,k))
for l in range(1,k):
    v = np.random.randn(p)
    v = v/LA.norm(v)
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

#####################################
# plotting fonts
#####################################
yspace = np.array([u for u in range(len(sigma))])
xspace = np.array([u for u in range(len(N))])
x_labels = xspace[::8]
y_labels = yspace[::8]
Nxlabels = N[::8]
Sylabels = linsigma[::8]



#####################################
# ORACLE
#####################################
MatErrOracle = [[0] * len(N) for i in range(len(sigma))]
for i in range(len(sigma)):
    s = sigma[i]
    for j in range(len(N)):
        n = N[j]
        MatErrOracle[i][j] = int(k*p*(s**2)/n >= Thresh)


#####################################
# Get the slope for Gram Recovery
#####################################

MatErr = np.loadtxt('ErrMatrix.txt')
MatErrPlot = (MatErr >= Thresh)
(row,col) = np.matrix(MatErrPlot).shape

FG = []
for c in range(len(N)):
    count = 0
    for r in range(len(sigma)-1,-1,-1):

        if MatErrPlot[r][c]:
            count += 1
        else:
            FG.append(row - count)
            break

xG = np.asarray([u for u in range(len(N))]).reshape(-1, 1)
yG = np.asarray(FG).reshape(-1, 1)

#fit the least squares and get the slope
reg_Gram   = linear_model.LinearRegression()
reg_Gram.fit(xG, yG)
slopeG = reg_Gram.coef_[0][0]

#####################################
# Get the slope for Oracle
#####################################

(row,col) = np.matrix(MatErrOracle).shape

FO = []
for c in range(len(N)):
    count = 0
    for r in range(len(sigma)-1,-1,-1):

        if MatErrOracle[r][c]:
            count += 1
        else:
            FO.append(row - count)
            break

    if r == 0:
        FO.append(row - count)

xO = np.asarray([u for u in range(len(N) - 8)]).reshape(-1, 1)
yO = np.asarray([FO[u] for u in range(len(FO)-8)]).reshape(-1, 1)

#fit the least squares and get the slope
reg_Oracle = linear_model.LinearRegression()
reg_Oracle.fit(xO, yO)
slopeO = reg_Oracle.coef_[0][0]




#####################################
# Coordinates for lines
#####################################
x_OR = [u for u in range(int(alpha*len(N)))]
y_OR = [FG[0] + slopeO*u for u in range(int(alpha*len(N)))]
x_GR = [u for u in range(int(len(N)))]
y_GR = [FG[OrdinatesOrigin] + 0.5*slopeO*u for u in range(int(len(N)))]


#####################################
# Actual plot
#####################################
fig, ax = plt.subplots(figsize = (12,8))
im = plt.imshow(MatErr, interpolation = 'none', cmap='binary',origin='lower')
plt.plot(x_GR,y_GR,'r',label = '$\sigma = c\cdot\sqrt{N}$')
plt.plot(x_OR,y_OR,'b',label = '$\sigma = C \cdot N^{1/4}$')
plt.xticks(x_labels,[str(u) for u in Nxlabels],fontsize=18)
plt.yticks(y_labels,['$2^{' + str(round(u, 1)) + '}$' for u in Sylabels],fontsize=18)
# plt.locator_params(axis = 'x',nbins=len(Nplot))
# plt.locator_params(axis = 'y',nbins=len(sigmaplot))
plt.title('Estimation Error ',fontsize=30)
plt.xlabel('Number of observations N',fontsize=24)
plt.ylabel('Noise level $\sigma$',fontsize=24)
plt.legend(loc='lower right',prop={'size': 20},frameon=False)
#plt.colorbar()
plt.show()
