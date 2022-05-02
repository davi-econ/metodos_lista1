import numpy as np
from numpy.random import normal
import scipy as sc
from scipy.ndimage.interpolation import shift
from scipy.stats import norm
import matplotlib.pyplot as plt
import array_to_latex as atl
import statsmodels.formula.api as smf
import pandas as pd
from stargazer.stargazer import Stargazer

random.seed(123)
mu = 0
rho = 0.95
sigma = 0.007


###### Espaço de Estados Exógenos ########

def disc_tauchen(rho, mu, sigma, m, n):
    # theta é o vetor de estados
    # definir primeiro os extremos;
    theta_1 = mu - m*sigma/np.sqrt(1-rho**2) #theta_0 se fosse coerente
    theta_n = mu + m*sigma/np.sqrt(1-rho**2)
    # todo o vetor de estados
    theta  = np.linspace(theta_1, theta_n, n)
    
    # Pontos entre cada ponto do grid + extremos (inf e -inf)
    midpoints = np.empty(n+1)
    midpoints[0] = -np.Inf
    midpoints[n] = np.Inf
    for i in np.arange(1,n,1):
        midpoints[i]=0.5*(theta[i-1]+theta[i])
    
    # matriz de transição vazia
    transition = np.empty((n, n))
    
    # preenchimento da matriz
    for i in range(n):
        for j in range(n):
            # como o vetor midpoint tem infinito nos extremos, a formulação a seguir engloba os casos dos extremos de j = 1 ou j = n 
            upper = midpoints[j+1] - (1-rho)*mu - rho*theta[i]
            lower = midpoints[j] - (1-rho)*mu - rho*theta[i]
            transition[i,j] = norm.cdf(upper/sigma, loc=0, scale=1)-norm.cdf(lower/sigma, loc=0, scale=1)

    return [theta, transition]

###### método de Rouwenhorst
def disc_rouwenhorst(rho, mu , sigma ,m , n ):
    p = (1 + rho)/2
    PN = np.array([[p, 1-p], [1-p, p]])
    # Adicionar linhas e colunas a partir da P2, mas aqui já denominada PN para o loop.
    for i in np.arange(3,10,1):
        # To pensando em algo análogo a quadrantes
        Xq1 = np.column_stack((PN,np.zeros(i-1)))
        Xq1 = np.row_stack((Xq1,np.zeros(i)))
    
        Xq2 = np.column_stack((np.zeros(i-1), PN))
        Xq2 = np.row_stack((Xq2,np.zeros(i)))

        Xq3 = np.column_stack((PN,np.zeros(i-1)))
        Xq3 = np.row_stack((np.zeros(i),Xq3))
    
        Xq4 = np.column_stack((np.zeros(i-1), PN))
        Xq4 = np.row_stack((np.zeros(i),Xq4))

        PN = p*Xq1 + (1-p)*Xq2 + (1-p)*Xq3 + + p*Xq4
    
    # theta é o vetor de estados
    # definir primeiro os extremos;
    # primeira o sigma diferente
    st = sigma/np.sqrt(1-rho**2)
    theta_1 = mu - np.sqrt(n-1)*st #theta_0 se fosse coerente
    theta_n = mu + np.sqrt(n-1)*st
    # todo o vetor de estados
    theta  = np.linspace(theta_1, theta_n, n)
    
    # coloquei a normalização das linhas aqui pra não confundir com o outro loop
    # para que os valores da linha somem 1:
    for i in range(PN.shape[1]):
        PN[i] = PN[i]/PN[i].sum()
    return(theta,PN)

theta_t,P_t = disc_tauchen(rho = 0.95, mu = 0, sigma = 0.007,m = 3, n = 9) 
theta_r, P_r = disc_rouwenhorst(rho = 0.95, mu = 0, sigma = 0.007,m = 3, n = 9)

theta_t2,P_t2 = disc_tauchen(rho = 0.99, mu = 0, sigma = 0.007,m = 3, n = 9) 
theta_r2,P_r2 = disc_rouwenhorst(rho = 0.99, mu = 0, sigma = 0.007,m = 3, n = 9) 


### Questão 3 ####
### erros do processo contínuo
erros = normal(0,0.007,10001)
z = np.zeros(10001)

## AR(1) contínuo
for i in np.arange(1,10001,1):
    z[i] = 0.95*z[i-1] + erros[i]


## Vetores para os processos discretizados
theta_s_t = np.zeros(10001)
theta_s_r = np.zeros(10001)
i_t, i_r = 4,4
## preenchimento dos processos discretizados usando os choques do processo contínuo

for i in np.arange(1,10001,1):
    i_t = np.where(norm.cdf(erros[i],loc = 0, scale=0.007) <= np.cumsum(P_t[i_t,:]))[0][0]
    i_r = np.where(norm.cdf(erros[i],loc = 0, scale=0.007) <= np.cumsum(P_r[i_r,:]))[0][0]
    theta_s_t[i] = theta_t[i_t]
    theta_s_r[i] = theta_r[i_r]

### Gráficos 
plt.title("Processos simulados com rho = 0.95")
plt.plot(np.arange(0,1001,1),z,label='Contínuo')
plt.plot(np.arange(0,1001,1),theta_s_t,label='Tauchen')
plt.plot(np.arange(0,1001,1),theta_s_r,label='Rouwenhorst')
plt.legend(loc = 'best')
#plt.show()

##### Questão 4 #####
### Refazendo tudo com rho = 0.99
### erros do processo contínuo
z2 = np.zeros(10001)

## AR(1) contínuo
for i in np.arange(1,10001,1):
    z2[i] = 0.99*z2[i-1] + erros[i]


## Vetores para os processos discretizados
theta_s_t2 = np.zeros(10001)
theta_s_r2 = np.zeros(10001)
i_t, i_r = 4,4
## preenchimento dos processos discretizados usando os choques do processo contínuo
for i in np.arange(1,10001,1):
    i_t = np.where(norm.cdf(erros[i],loc = 0, scale=0.007) <= np.cumsum(P_t2[i_t,:]))[0][0]
    i_r = np.where(norm.cdf(erros[i],loc = 0, scale=0.007) <= np.cumsum(P_r2[i_r,:]))[0][0]
    theta_s_t2[i] = theta_t2[i_t]
    theta_s_r2[i] = theta_r2[i_r]

    
### Gráficos 
plt.title("Processos simulados com rho = 0.99")
plt.plot(np.arange(0,1001,1),z2,label='Contínuo')
plt.plot(np.arange(0,1001,1),theta_s_t2,label='Tauchen')
plt.plot(np.arange(0,1001,1),theta_s_r2,label='Rouwenhorst')
plt.legend(loc = 'best')
#plt.show()

## 
data_reg_95 = pd.DataFrame({"AR1":z, "Tauchen":theta_s_t, "Rouwenhorst":theta_s_r})
data_reg_99 = pd.DataFrame({"AR1":z2, "Tauchen":theta_s_t2, "Rouwenhorst":theta_s_r2})


reg_t_95 = smf.ols("theta_s_t ~ shift(theta_s_t, -1) - 1", data = data_reg_95).fit()
reg_r_95 = smf.ols("theta_s_r ~ shift(theta_s_r, -1) - 1", data = data_reg_95).fit()

reg_t_99 = smf.ols("theta_s_t2 ~ shift(theta_s_t2, -1) - 1", data = data_reg_99).fit()
reg_r_99 = smf.ols("theta_s_r2 ~ shift(theta_s_r2, -1) - 1", data = data_reg_99).fit()