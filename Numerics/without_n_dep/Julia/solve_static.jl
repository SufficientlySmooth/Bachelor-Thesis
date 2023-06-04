#Path of this file: "C:/Users/Jan-Philipp/Documents/Eigene Dokumente/Physikstudium/6. Semester/Bachelor-Thesis/Numerics/without_n_dep/Julia/solve_static.jl"
import NPZ
import DifferentialEquations
import LinearAlgebra
import ThreadsX
using NPZ,DifferentialEquations, LinearAlgebra, ThreadsX

N = 200
n = 1000 #for how many time steps we want to save the solution

name = "eta=1.111,N=200.npy"
flat = npzread("C:/Users/Jan-Philipp/Documents/Eigene Dokumente/Physikstudium/6. Semester/Bachelor-Thesis/Numerics/without_n_dep/quadratic Hamiltonians N=200, different etas/"*name)
#print(flat)
function flat_arr(om,V,W,eps)
    return vcat(vec(om),vec(V),vec(W),eps)
end

function unpack_arr(flat,N)
    N = Int(N)
    om0 = flat[1:N]
    V0 = flat[N+1:N+Int(N^2)]
    W0 = flat[N+Int(N^2)+1:N+Int(2*N^2)]
    eps = flat[end]
    V = reshape(V0,(N,N))
    W = reshape(W0,(N,N))
    return om0,V,W,eps
end

function f(u,p,t)
    N = Int(p[1]/2)
    om,V,W,eps = unpack_arr(u,N)
    print(t,"\n")
    Wdag = conj(W)
    
    om_ret = Float64.(ThreadsX.collect(1:N))
    ThreadsX.map((k)->ThreadsX.sum([2*V[q,Int(k)]*V[Int(k),q]*(om[Int(k)]-om[q])-2*(W[Int(k),q]+W[q,Int(k)])*(om[Int(k)]+om[q])*(Wdag[q,Int(k)]+Wdag[Int(k),q]) for q in 1:N]),om_ret)
    print(om_ret)
    #om_ret = [ThreadsX.sum([2*V[q,k]*V[k,q]*(om[k]-om[q])-2*(W[k,q]+W[q,k])*(om[k]+om[q])*(Wdag[q,k]+Wdag[k,q]) for q in 1:N]) for k in 1:N]
    
    V_ret = Float64.([(p_,p) for p_ in 1:N for p in 1:N])
    ThreadsX.mapreduce(((q,q_),)->-V[Int(q),Int(q_)]*(om[Int(q)]-om[Int(q_)])^2+ThreadsX.sum([-(W[Int(q),p]+W[p,Int(q)])*(Wdag[p,Int(q_)]+Wdag[Int(q_),p])*(om[Int(q)]+om[Int(q_)]+2*om[p])+V[p,Int(q_)]*V[Int(q),p]*(om[Int(q)]+om[Int(q_)]-2*om[p]) for p in 1:N if p ∉ (Int(q),Int(q_))]),+,V_ret) #maybe need to change q,q_
    V_ret = reshape(V_ret,(N,N)).*(ones(N,N)-Diagonal(ones(N)))
    
    W_ret = Float64.([(p_,p) for p_ in 1:N for p in 1:N])
    ThreadsX.mapreduce(((p,p_),)->-W[Int(p),Int(p_)]*(om[Int(p)]+om[Int(p_)])^2+ThreadsX.sum([-V[Int(p),q]*(om[q]+om[Int(p_)])*(W[Int(p_),q]+W[q,Int(p_)])+V[Int(p),q]*(om[Int(p)]-om[q])*(W[q,Int(p_)]+W[Int(p_),q]) for q in 1:N if q!=Int(p)]),+,W_ret)
    W_ret = reshape(W_ret,(N,N))

    eps_ret = Float64.([(p,p_) for p in 1:N for p_ in 1:N])
    ThreadsX.mapreduce(((p,p_),)->-2*(W[Int(p),Int(p_)]+W[Int(p_),Int(p)])*(om[Int(p)]+om[Int(p_)])*Wdag[Int(p),Int(p_)],+,eps_ret)
    
    ret = flat_arr(om_ret,V_ret,W_ret,eps_ret)
    return ret
end

om0,V0,W0,eps0 = unpack_arr(flat,N)
om0 = om0 + diag(V0)
V0 = V0 - Diagonal(diag(V0))

V11 = V0[1:Int(N/2),1:Int(N/2)] #use symmetry of the problem and only calculate 1/4th of the matrix elements
W11 = V0[1:Int(N/2),1:Int(N/2)] #use symmetry of the problem and only calculate 1/4th of the matrix elements
om1 = om0[1:Int(N/2)]

u0 = flat_arr(om1,V11,W11,eps0)

tspan = (0.0,30)

prob = ODEProblem(f,u0,tspan,[N])
sol = solve(prob, Tsit5(),saveat=LinRange(tspan[1],tspan[2],n))

npzwrite("C:/Users/Jan-Philipp/Documents/Eigene Dokumente/Physikstudium/6. Semester/Bachelorarbeit_sol_files/N=200,different etas/"*"sol_"*name,sol)
npzwrite("C:/Users/Jan-Philipp/Documents/Eigene Dokumente/Physikstudium/6. Semester/Bachelorarbeit_sol_files/N=200,different etas/"*"sol_t"*name,sol.t)