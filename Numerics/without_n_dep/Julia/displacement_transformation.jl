#Path of this file: "C:/Users/Jan-Philipp/Documents/Eigene Dokumente/Physikstudium/6. Semester/Bachelor-Thesis/Numerics/without_n_dep/Julia/displacement_transformation.jl"
import NPZ, NLsolve
using NPZ, NLsolve

PATH = "C:/Users/Jan-Philipp/Documents/Eigene Dokumente/Physikstudium/6. Semester/Bachelorarbeit Quadratic Hamiltonians/"

function om(k,consts)
    xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L = consts
    return  cc*abs(k)*sqrt(1+(k^2*xi^2)/2)
end

function W(k,consts)
    xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L = consts
    return (k^2*xi^2/(2+k^2*xi^2))^(1/4)
end

function s(k,consts)
    xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L = consts
    w = W(k,consts)
    return 1/2*(w+1/w)
end

function c(k,consts)
    xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L = consts
    w = W(k,consts)
    return 1/2*(1/w-w)
end

function gen_1Dgrid(consts)
    xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L = consts
    n_min = Int(floor(lamb_IR/dk))
    n_max = Int(ceil(lamb_UV/dk))
    k_pos = collect(n_min:n_max)*dk
    k_neg = -k_pos[end:-1:1]
    return vcat(k_neg,k_pos)
end

function V0(k,k_,consts)
    xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L = consts
    return 2*pi*g_ib/L*(c(k,consts)*c(k_,consts)+s(k,consts)*s(k_,consts))
end

function W0(k,k_,consts)
    xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L = consts
    return -2*pi*g_ib/L*s(k,consts)*c(k_,consts)
end

function eps0(grid,consts)
    xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L = consts
    g_ib*n0+2*pi*g_ib/L*sum([s(k,consts)^2 for k in grid])
end

function omega0(k,consts)
    xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L = consts
    return 2*pi/L*om(k,consts)
end

function W0_tilde(k,consts)
    xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L = consts
    return  g_ib*sqrt(n0)/sqrt(2*pi)*(2*pi/L)^(3/2)*W(k,consts)
end

function omega0_arr(grid,consts)
    xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L = consts
    return [omega0(k,consts) for k in grid]
end

function W0_tilde_arr(grid,consts)
    xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L = consts
    return [W0_tilde(k,consts) for k in grid]
end

function W0_arr(grid,consts)
    xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L = consts
    N = length(grid)
    return reshape([W0(k,k_,consts) for k_ in grid for k in grid],(N,N))
end

function V0_arr(grid,consts)
    xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L = consts
    N = length(grid)
    return reshape([V0(k,k_,consts) for k_ in grid for k in grid],(N,N))
end

function func!(F,alphas,grid,consts) # linear part which we will minimize
    xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L = consts
    N = length(grid)
    function alpha_(index) # returns alpha(-k)
        return alphas[N - index + 1]
    end
    for i in 1:N
        k = grid[i]
        F[i] = sum([V0(k,grid[j],consts)*alphas[j]+W0(grid[j],k,consts)*conj(alpha_(j))+W0(-k,grid[j],consts)*conj(alphas[j]) for j in 1:N]) + W0_tilde(-k,consts)
    end
end

function remove_linear(grid,consts,tol = 1e-10)
    xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L = consts
    alpha0 = zeros(length(grid))
    sol = nlsolve((F,alphas)->func!(F,alphas,grid,consts),alpha0, ftol = tol)
    return sol.zero
end

function test_convergence(alphas,grid,consts)
    xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L = consts
    N = length(grid)
    function alpha_(index) # returns alpha(-k)
        return alphas[N - index + 1]
    end
    for i in 1:N
        k = grid[i]
        print(sum([V0(k,grid[j],consts)*alphas[j]+W0(grid[j],k,consts)*conj(alpha_(j))+W0(-k,grid[j],consts)*conj(alphas[j]) for j in 1:N]) + W0_tilde(-k,consts),"\n")
        print(sum([V0(grid[j],k,consts)*conj(alphas[j])+W0(-k,grid[j],consts)*alpha_(j)+W0(grid[j],k,consts)*alpha_(j) for j in 1:N]) + W0_tilde(k,consts),"\n")

    end
end

function get_quadratic_Hamiltonian(grid,consts)
    xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L = consts
    N = length(grid)
    alphas = remove_linear(grid,consts)
    function alpha_(index) # returns alpha(-k)
        return alphas[N - index + 1]
    end
    eps_help_arr = [W0(grid[i],grid[j],consts)*(alpha_(i)*alphas[j]+conj(alpha_(i)*alphas[j]))+V0(grid[i],grid[j],consts)*conj(alphas[i]*alphas[j]) for i in 1:N for j in 1:N]
    eps = eps0(grid,consts) + sum([W0_tilde(grid[i],consts)*(alphas[i]+conj(alpha_(i))) for i in 1:N]) +sum(eps_help_arr)
    return omega0_arr(grid,consts),V0_arr(grid,consts),W0_arr(grid,consts),eps
end

for eta in LinRange(-5,5,20)
    xi = 1 #characteristic length (BEC healing length)
    cc = 1 #c in free Bogoliubov dispersion (speed of sound)
    lamb_IR = 1e-1 #Infrared cutoff
    lamb_UV = 1e1 #ultra-violet cutoff
    m_b = 1/(sqrt(2)*cc*xi) #reduced mass = boson mass in the limit of infinite impurity mass
    #eta = 1 #will be varied between -10...10 later
    n0 = 1.05/xi #
    gamma = 0.438
    a_bb = 2/(n0*gamma)
    g_bb = -2/(m_b*a_bb)
    g_ib = eta*g_bb
    dk = 1e-1
    L = 2*pi/dk
    consts = (xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L)

    grid = gen_1Dgrid(consts)
    om_ret, V_ret, W_ret, eps_ret = get_quadratic_Hamiltonian(grid,consts)
   
    ret = vcat(vec(om_ret),vec(V_ret),vec(W_ret),eps_ret)
   
    npzwrite(PATH*"Ham_eta="*string(eta)*",N="*string(length(grid))*",lambda_IR="*string(lamb_IR)*",lambda_UV="*string(lamb_UV)*".npy",ret)
end


