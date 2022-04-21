
from secrets import choice
from shutil import which
import numpy as np
from numpy.random import normal
import scipy as sc
from scipy.stats import norm
import matplotlib.pyplot as plt
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


theta_t,P_t = disc_tauchen(rho = 0.95, mu = 0, sigma = 0.007,m = 3, n = 9) 
theta_t

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

theta_r, P_r = disc_rouwenhorst(rho = 0.95, mu = 0, sigma = 0.007,m = 3, n = 9)
theta_r


### Questão 3 ####
### erros do processo contínuo
erros = normal(0,0.007,1001)
z = np.zeros(1001)

## AR(1) contínuo
for i in np.arange(1,1001,1):
    z[i] = rho*z[i-1] + erros[i]

plt.plot(np.arange(0,1001,1),z)
plt.show()

## Vetores para os processos discretizados
theta_s_t = np.zeros(1001)
theta_s_r = np.zeros(1001)

## preenchimento dos processos discretizados usando os choques do processo contínuo
for i in np.arange(1,1001,1):
    i_t = np.argmin(abs(theta_t - z[i-1]))
    i_r = np.argmin(abs(theta_r - z[i-1]))
    theta_s_t[i] = np.random.choice(theta_t,1,False,p = P_t[i_t])
    theta_s_r[i] = np.random.choice(theta_r,1,False,p = P_r[i_r])

### Gráficos 
plt.plot(np.arange(0,1001,1),z,label='Contínuo')
plt.plot(np.arange(0,1001,1),theta_s_t,label='Tauchen')
plt.plot(np.arange(0,1001,1),theta_s_r,label='Rouwenhorst')
plt.legend(loc = 'best')
plt.show()

##### Questão 4 #####

#z_r = np.zeros(1001)
#z_t = np.zeros(1001)
#for i in range(1000):
#    if i == 0:
#        z_r[i] = 0
#        z_t[i] = 0
#    else:
#        z_r[i] = np.random.choice(theta_r,1,False,p = P_r[np.where(z_r[i-1] == theta_r)[0][0]])
#        z_t[i] = np.random.choice(theta_t,1,False,p = P_t[np.where(z_t[i-1] == theta_t)[0][0]])


z_t_lag = theta_s_t[0:1000]
z_t = theta_s_t[1:1001]

z_r_lag = theta_s_r[0:1000]
z_r = theta_s_r[1:1001]

rho_estimado_tauch = (z_t @ z_t_lag)/(z_t_lag.T @ z_t_lag)
rho_estimado_rouw = (z_r @ z_r_lag)/(z_r_lag.T @ z_r_lag)


### Refazendo tudo com rho = 0.99

theta_t,P_t = disc_tauchen(rho = 0.99, mu = 0, sigma = 0.007,m = 3, n = 9) 
theta_r,P_r = disc_rouwenhorst(rho = 0.99, mu = 0, sigma = 0.007,m = 3, n = 9) 

rho = 0.99
### erros do processo contínuo
erros = normal(0,0.007,1001)
z = np.zeros(1001)

## AR(1) contínuo
for i in np.arange(1,1001,1):
    z[i] = rho*z[i-1] + erros[i]

plt.plot(np.arange(0,1001,1),z)
plt.show()

## Vetores para os processos discretizados
theta_s_t = np.zeros(1001)
theta_s_r = np.zeros(1001)

## preenchimento dos processos discretizados usando os choques do processo contínuo
for i in np.arange(1,1001,1):
    i_t = np.argmin(abs(theta_t - z[i-1]))
    i_r = np.argmin(abs(theta_r - z[i-1]))
    theta_s_t[i] = np.random.choice(theta_t,1,False,p = P_t[i_t])
    theta_s_r[i] = np.random.choice(theta_r,1,False,p = P_r[i_r])

### Gráficos 
plt.plot(np.arange(0,1001,1),z,label='Contínuo')
plt.plot(np.arange(0,1001,1),theta_s_t,label='Tauchen')
plt.plot(np.arange(0,1001,1),theta_s_r,label='Rouwenhorst')
plt.legend(loc = 'best')
plt.show()


## compatibilizando a dimensão do lag com com a dimensão do tempo presente
z_t_lag = theta_s_t[0:1000]
z_t = theta_s_t[1:1001]

## 
z_r_lag = theta_s_r[0:1000]
z_r = theta_s_r[1:1001]

(z_t @ z_t_lag)/(z_t_lag.T @ z_t_lag)
(z_r @ z_r_lag)/(z_r_lag.T @ z_r_lag)

