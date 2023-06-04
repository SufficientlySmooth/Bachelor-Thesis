#Path of this file: "C:/Users/Jan-Philipp/Documents/Eigene Dokumente/Physikstudium/6. Semester/Bachelor-Thesis/Numerics/without_n_dep/Julia/solve.jl"
#Path of this file: "C:/Users/Jan-Philipp/Documents/Eigene Dokumente/Physikstudium/6. Semester/Bachelor-Thesis/Numerics/without_n_dep/Julia/solve_static.jl"
import NPZ
import DifferentialEquations
import LinearAlgebra
using NPZ,DifferentialEquations, LinearAlgebra

N = 200
n = 1000 #for how many time steps we want to save the solution

name = "eta=2.929,N=200.npy"
flat = npzread("C:/Users/Jan-Philipp/Documents/Eigene Dokumente/Physikstudium/6. Semester/Bachelor-Thesis/Numerics/without_n_dep/quadratic Hamiltonians N=200, different etas/"*name)


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
    print(t,"\n")
    ret = Vector{Float64}(undef,N+2*N^2+1)  
    for k in 1:N
        ret[k] = sum([2*u[N*q+k]*u[N*k+q]*(u[k]-u[q])-2*(u[N^2+N*k+q]+u[N^2+N*q+k])*(u[k]+u[q])*(conj(u[N^2+N*q+k])+conj(u[N^2+N*k+q])) for q in 1:N])
    end
    for i in (N+1):(N+N^2)
        q_ = mod(i,N)
        if q_==0
            q_=N 
        end
        q = Int((i - q_)/N) 

        if q==0
            q = N 
        end

        if q==q_
            ret[i] = 0
        else
            ret[i] = -u[N*q+q_]*(u[q]-u[q_])^2+sum([-(u[N^2+N*(q)+p]+u[N^2+N*p+(q)])*(conj(u[N^2+N*p+(q_)])+conj(u[N^2+N*(q_)+p]))*(u[(q)]+u[(q_)]+2*u[p])+u[N*p+(q_)]*u[N*(q)+p]*(u[(q)]+u[(q_)]-2*u[p]) for p in 1:N if p ∉ (q,q_)])
        end
    end
    for i in (N+N^2+1):(N+2*N^2)

        p_ = mod(i-N^2,N)
        if p_==0
            p_=N 
        end
        p = Int((i - N^2 - p_)/N) 

        if p==0
            p = N 
        end

        ret[i] = -u[N^2+N*p+p_]*(u[p]+u[p_])^2+sum([-u[N*p+q]*(u[q]+u[p_])*(u[N^2+N*p_+q]+u[N^2+N*q+p_])+u[N*p+q]*(u[p]-u[q])*(u[N^2+N*q+p_]+u[N^2+N*p_+q]) for q in 1:N if q!=p])
    end   
    ret[end] = -2*sum([(u[N^2+N*p+p_]+u[N^2+N*p_+p])*(u[p]+u[p_])*conj(u[N^2+N*p+p_]) for p in 1:N for p_ in 1:N])

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
sol = solve(prob, Tsit5(), reltol = 1e-8, abstol = 1e-8,saveat=LinRange(tspan[1],tspan[2],n))

npzwrite("C:/Users/Jan-Philipp/Documents/Eigene Dokumente/Physikstudium/6. Semester/Bachelorarbeit_sol_files/N=200,different etas/"*"sol_"*name,sol)
npzwrite("C:/Users/Jan-Philipp/Documents/Eigene Dokumente/Physikstudium/6. Semester/Bachelorarbeit_sol_files/N=200,different etas/"*"sol_t_"*name,sol.t)