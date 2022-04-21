using Distributions, Random, Plots

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
theta_t
theta_r