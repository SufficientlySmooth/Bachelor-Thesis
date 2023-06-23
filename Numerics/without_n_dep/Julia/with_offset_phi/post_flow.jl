#Path of this file: "C:/Users/Jan-Philipp/Documents/Eigene Dokumente/Physikstudium/6. Semester/Bachelor-Thesis/Numerics/without_n_dep/Julia/with_offset_phi/post_flow.jl"
import LinearAlgebra, NPZ, Base, Plots
using LinearAlgebra, NPZ, Plots

PATH  = "C:/Users/Jan-Philipp/Documents/Eigene Dokumente/Physikstudium/6. Semester/Bachelorarbeit_sol_files/N=40,different etas_full_with_phi/"

flat = []

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

for filename in readdir(PATH)[1:1]
    global a,b
    eta_str = split(split(filename,'=')[2],',')[1]
    eta = parse(Float64,eta_str)
    print("eta = ",eta_str,"\n")
    filepath = PATH*filename
    #print(filepath*"\n")
    a,b = npzread(filepath)   
end