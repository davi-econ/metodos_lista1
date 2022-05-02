using Distributions, Random, Plots
using LinearAlgebra
using StatsBase
using NBInclude
using DataFrames
using GLM

# Carregando as funções que discretizam o processo, gerando o grid de estados e a matriz de transição
@nbinclude("tauchen.ipynb")
@nbinclude("rouwenhorst.ipynb")
@nbinclude("simulacao_processos.ipynb")

#### Questões 1 e 2
# obtendo valores iniciais
theta_t1,P_t1 = disc_t(9;mu =  0,sigma = 0.007,rho =  0.95,m = 3)
theta_r1,P_r1 = disc_r( 9; mu = 0, sigma = 0.007, rho = 0.95,m = 3)

theta_t2,P_t2 = disc_t(9;mu =  0,sigma = 0.007,rho =  0.95,m = 3)
theta_r2,P_r2 = disc_r( 9; mu = 0, sigma = 0.007, rho = 0.95,m = 3)

#### Questão 3 ####
# simulacao para O processo contínuo, via tauchen e via rouwenhorst
z1, z_t1, z_r1 = simulacao_processos(10000;rho = 0.95)
z2, z_t2, z_r2 = simulacao_processos(10000;rho = 0.99)

# Gráficos
# apesar da simulação ter sido para 10000 erros, os gráficos mostram somente os 1000 primeiros para ficar mais legível
p095 = plot(1:1000,z1[1:1000,], label = "AR1 contínuo", title = "Processos simulados rho = 0.95")
p095 = plot!(1:1000,z_t1[1:1000,], label = "Tauchen")
p095 = plot!( 1:1000,z_r1[1:1000,], label = "Rouwenhorst")

p099 = plot(1:1000,z2[1:1000,], label = "AR1 contínuo", title = "Processos simulados rho = 0.99")
p099 = plot!(1:1000,z_t2[1:1000,], label = "Tauchen")
p099 = plot!( 1:1000,z_r2[1:1000,], label = "Rouwenhorst")
##### Questão 4 #####
## organizando entre nível e lag
preg95 = DataFrame(Tauchen = z_t1,Rouw = z_r1,lt =lag(z_t1),lr = lag(z_r1))
preg99 = DataFrame(Tauchen = z_t2,Rouw = z_r2,lt =lag(z_t2),lr = lag(z_r2))
# regressao
tauc = lm(@formula(Tauchen ~ lt + 0 ),preg95)
rouw = lm(@formula(Rouw ~ lr + 0 ),preg99)

