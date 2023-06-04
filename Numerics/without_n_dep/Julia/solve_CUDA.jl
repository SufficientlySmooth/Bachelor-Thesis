import NPZ
import DifferentialEquations
import LinearAlgebra, CUDA
using NPZ,DifferentialEquations, LinearAlgebra, CUDA

N = 200

flat = cu(npzread("C:/Users/Jan-Philipp/Documents/Eigene Dokumente/Physikstudium/6. Semester/Bachelor-Thesis/Numerics/without_n_dep/quadratic Hamiltonians N=200, different etas/eta=1.111,N=200.npy"))
print(flat)

function flat_arr(om,V,W,eps)
    return cu(vcat(vec(om),vec(V),vec(W),eps))
end

function unpack_arr(flat,N)
    N = Int(N)
    om0 = flat[1:N]
    V0 = flat[N+1:N+Int(N^2)]
    W0 = flat[N+Int(N^2)+1:N+Int(2*N^2)]
    eps = flat[end]
    V = reshape(V0,(N,N))
    W = reshape(W0,(N,N))
    return cu(om0),cu(V),cu(W),cu(eps)
end

function f(u,p,t)
    N = p[1]
    om,V,W,eps = unpack_arr(u,N)
    print(t,"\n")
    Wdag = conj(W)
    
    om_ret = [sum([2*V[q,k]*V[k,q]*(om[k]-om[q])-2*(W[k,q]+W[q,k])*(om[k]+om[q])*(Wdag[q,k]+Wdag[k,q]) for q in 1:N]) for k in 1:N]
    
    V_ret = reshape([-V[q,q_]*(om[q]-om[q_])^2+sum([-(W[q,p]+W[p,q])*(Wdag[p,q_]+Wdag[q_,p])*(om[q]+om[q_]+2*om[p])+V[p,q_]*V[q,p]*(om[q]+om[q_]-2*om[p]) for p in 1:N if p ∉ (q,q_)]) for q_ in 1:N for q in 1:N],(N,N)).*(ones(N,N)-Diagonal(ones(N)))

    W_ret = reshape([-W[p,p_]*(om[p]+om[p_])^2+sum([-V[p,q]*(om[q]+om[p_])*(W[p_,q]+W[q,p_])+V[p,q]*(om[p]-om[q])*(W[q,p_]+W[p_,q]) for q in 1:N if q!=p]) for p_ in 1:N for p in 1:N],(N,N))

    eps_ret = -2*sum([(W[p,p_]+W[p_,p])*(om[p]+om[p_])*Wdag[p,p_] for p in 1:N for p_ in 1:N])
    
    ret=flat_arr(om_ret,V_ret,W_ret,eps_ret)
    return ret
end

om0,V0,W0,eps0 = unpack_arr(flat,N)
om0 = om0 + diag(V0)
V0 = V0 - Diagonal(diag(V0))

u0 = flat_arr(om0,V0,W0,eps0)

tspan = (0.0,30)

prob = ODEProblem(f,u0,tspan,[N])
sol = solve(prob)#, Tsit5(), reltol = 1e-8, abstol = 1e-8)

npzwrite("sol_abc.npy",sol)
npzwrite("sol_abc_t.npy",sol.t)
