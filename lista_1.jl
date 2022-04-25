using Distributions, Random, Plots
using LinearAlgebra
using StatsBase
using NBInclude

# Carregando as funções que discretizam o processo, gerando o grid de estados e a matriz de transição
@nbinclude("tauchen.ipynb")
@nbinclude("rouwenhorst.ipynb")
@nbinclude("simulacao_processos.ipynb")

#### Questões 1 e 2
# obtendo valores iniciais
theta_t,P_t = disc_t( 9, 0, 0.007, 0.95, 3)
theta_r,P_r = disc_r( 9, 0, 0.007, 0.95, 3)

#### Questão 3 ####
# simulacao para O processo contínuo, via tauchen e via rouwenhorst
z, z_t, z_r = simulacao_processos(10000)
# Gráficos
# apesar da simulação ter sido para 10000 erros, os gráficos mostram somente os 1000 primeiros para ficar mais legível
plot(1:1000,z[1:1000,], label = "AR1 contínuo", title = "Processos simulados")
plot!(1:1000,z_r[1:1000,], label = "Tauchen")
plot!(1:1000,z_t[1:1000,], label = "Rouwenhorst")



##### Questão 4 #####
## organizando entre nível e lag
# de tauchen
z_t_lag = z_t[1:9999,1]
z_t = z_t[2:10000,1]

# de Rouwenhorst
z_r_lag = z_r[1:9999,1]
z_r = z_r[2:10000,1]

# encontrando rho a partir dos dados simulados
inv(z_t_lag'*z_t_lag)*(z_t_lag'*z_t)
inv(z_r_lag'*z_r_lag)*(z_r_lag'*z_r)

### Questão 5
# refazer para rho = 0.99
z, z_r,z_t = simulacao_processos(10000;rho = 0.99,mu = 0, sigma = 0.007,n = 9, m =3)
plot(z, label = "AR1 contínuo", title = "Processos simulados")
plot!(z_t, label = "Tauchen")
plot!(z_r, label = "Rouwenhorst")


z_t_lag = z_t[1:9999,1]
z_t = z_t[2:10000,1]

z_r_lag = z_r[1:9999,1]
z_r = z_r[2:10000,1]

inv(z_t_lag'*z_t_lag)*(z_t_lag'*z_t)
inv(z_r_lag'*z_r_lag)*(z_r_lag'*z_r)

