using Distributions, Random, Plots
using LinearAlgebra

## Discretização via Tauchen
function disc_t(N, mu, sigma, rho, m)
    # extremos e vetor de thetas
    theta_n = mu + sigma*m/sqrt(1-rho^2)
    theta_1 = mu - sigma*m/sqrt(1-rho^2)
    thetas = LinRange(theta_1,theta_n,N)

    # pontos entre cada theta e extremos
    midpoints = zeros(N+1)
    midpoints[1] = -Inf
    midpoints[N+1] = Inf
    # preenchimento destes
    for i in 2:N
        midpoints[i]=0.5*(thetas[i-1]+thetas[i])    
    end #loop intermediario

    # matriz de transição vazia
    P = zeros(N,N)
    # Preenchimento dela
    d = Normal(mu,sigma) # que distribuição vai usar
    for i in 1:N
        for j in 1:N
            upper = midpoints[j+1] - (1-rho)*mu - rho*thetas[i]
            lower = midpoints[j] - (1-rho)*mu - rho*thetas[i]
            P[i,j] = cdf(d,upper) - cdf(d,lower)
        end #colunas matriz transição
    end #linhas matriz transição
    return thetas, P
end #função tauchen 

## Discretização via Rouwenhorst
function disc_r(N, mu, sigma, rho, m)
    p = (1 + rho)/2
    # matriz inicial
    PN = [p 1-p;1-p p]
    # Adcionar linhas e colunas a partir de P2 = PN inicial
    # Pensando em quadrantes
    for i in 3:N
        Xq1 = hcat(PN,zeros(i-1))
        Xq1 = vcat(Xq1,zeros(1,i))

        Xq2 = hcat(zeros(i-1), PN)
        Xq2 = vcat(Xq2,zeros(1,i))

        Xq3 = hcat(PN,zeros(i-1))
        Xq3 = vcat(zeros(1,i),Xq3)

        Xq4 = hcat(zeros(i-1), PN)
        Xq4 = vcat(zeros(1,i),Xq4)

        PN = p.*Xq1 +(1-p).*Xq2 + (1-p).*Xq3 + p.*Xq4
    end #loop de preenchimento

    # definir theta com o desvio padrao adequado
    st = sigma/sqrt(1-rho^2)
    theta_1 = mu - st*sqrt(N-1)
    theta_N = mu + st*sqrt(N-1)
    theta = LinRange(theta_1,theta_N,N)

    # normalização das linhas para que somem 1
    for i in 1:N
        PN[i,:] = PN[i,:]/sum(PN[i,:])
    end # normalização
    
    return theta, PN
end



theta_t,P_t = disc_t( 9, 0, 0.007, 0.95, 3)
theta_r,P_r = disc_r( 9, 0, 0.007, 0.95, 3)

#### Questão 3 ####
seed = MersenneTwister(35)
erros = 0.007.*randn(seed,Float64,(1001,1))
z = zeros(1001,1)

# AR(1) contínuo
rho = 0.95
for i in 2:1001
    z[i,1] = rho*z[i-1] + erros[i,1]
end 


## Vetores para os processos discretizados
theta_s_t = zeros(1001,1)
theta_s_r = zeros(1001,1)

broadcast(abs, theta_r .- z[1])
argmin(abs.(theta_r .- z[1]))
## preenchimento dos processos discretizados usando os choques do processo contínuo
for i in 2:1001
    i_t = argmin(abs.(theta_t .- z[i-1]))
    i_r = argmin(abs.(theta_r .- z[i-1]))
    theta_s_t[i] = sample(theta_t, Weights(P_t[i_t,:]))
    theta_s_r[i] = sample(theta_r, Weights(P_r[i_r,:]))
end


plot(1:1001,z)
plot!(1:1001,theta_s_r)
plot!(1:1001,theta_s_t)


##### Questão 4 #####
## organizando entre lags e presente
# de tauchen
z_t_lag = theta_s_t[1:1000,1]
z_t = theta_s_t[2:1001,1]

# de Rouwenhorst
z_r_lag = theta_s_r[1:1000,1]
z_r = theta_s_r[2:1001,1]

# encontrando rho a partir dos dados simulados
inv(z_t_lag'*z_t_lag)*(z_t_lag'*z_t)
inv(z_r_lag'*z_r_lag)*(z_r_lag'*z_r)

### Questão 5
# refazer para rho = 0.99
rho = 0.99
theta_t,P_t = disc_t( 9, 0, 0.007, rho, 3)
theta_r,P_r = disc_r( 9, 0, 0.007, rho, 3)

erros = 0.007.*randn(seed,Float64,(1001,1))
z = zeros(1001,1)

for i in 2:1001
    z[i,1] = rho*z[i-1] + erros[i,1]
end 

theta_s_t = zeros(1001,1)
theta_s_r = zeros(1001,1)


for i in 2:1001
    i_t = argmin(abs.(theta_t .- z[i-1]))
    i_r = argmin(abs.(theta_r .- z[i-1]))
    theta_s_t[i] = sample(theta_t, Weights(P_t[i_t,:]))
    theta_s_r[i] = sample(theta_r, Weights(P_r[i_r,:]))
end


plot(1:1001,z)
plot!(1:1001,theta_s_r)
plot!(1:1001,theta_s_t)


z_t_lag = theta_s_t[1:1000,1]
z_t = theta_s_t[2:1001,1]

z_r_lag = theta_s_r[1:1000,1]
z_r = theta_s_r[2:1001,1]

inv(z_t_lag'*z_t_lag)*(z_t_lag'*z_t)
inv(z_r_lag'*z_r_lag)*(z_r_lag'*z_r)