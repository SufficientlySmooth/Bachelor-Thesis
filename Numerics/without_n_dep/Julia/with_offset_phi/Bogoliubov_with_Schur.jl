#Path of this file: "C:/Users/Jan-Philipp/Documents/Eigene Dokumente/Physikstudium/6. Semester/Bachelor-Thesis/Numerics/without_n_dep/Julia/with_offset_phi/Bogoliubov_with_Schur.jl"
import LinearAlgebra, NPZ, Base, Plots
using LinearAlgebra, NPZ, Plots

N = 40
PATH = ""
Omega = [Diagonal(ones(N)) zeros(N,N); zeros((N,N)) -Diagonal(ones(N))]
theta = [zeros((N,N)) Diagonal(ones(N)); Diagonal(ones(N)) zeros((N,N))] #missing complex conjugation operator!

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

function get_paired_eigenvalues(V,W)
    A = V
    B = 2*W 
    H = [A B; -conj(B) -conj(A)]
    sr = schur(H)
    eigvals = diag(sr.Schur)
    sortind = sortperm(Float64.(real(eigvals)))
    sorted_vals = eigvals[sortind]
    eigvals_ord = collect(zip(sorted_vals[1:Int(length(eigvals)/2)],(sorted_vals[Int(length(eigvals)/2)+1:length(eigvals)])[end:-1:1]))
    return eigvals_ord, H
end

function classify_eigenvalues(eigvals_ord,H)
    print(length(eigvals_ord))
    imag_eigvals = zeros(length(eigvals_ord))
    lambdas = zeros(length(eigvals_ord))
    lambdas_conj = zeros(length(eigvals_ord))
    for (i,(val1,val2)) in enumerate(eigvals_ord)
        vec1 = vec(nullspace(H-val1*Diagonal(ones(2*length(eigvals_ord)))))
        vec2 = vec(nullspace(H-val2*Diagonal(ones(2*length(eigvals_ord)))))
        
        if length(vec1) > 0 && length(vec2)>0
            mat_el1 = Float64(real(conj(Transpose(vec1))*Omega*vec1))
            mat_el2 = Float64(real(conj(Transpose(vec2))*Omega*vec2))
            #if abs(mat_el0)<0.9
            #    print("Matrix Element 1: ", mat_el0, " Matrix Element 2: ", mat_el1, "\n")
            #    print("Eigenvalues", eigvals_ord[i],"\n ...................................\n")
            #end
            if mat_el1>0 && mat_el2<0
                lambdas[i] = val1
                #if real(eigvals_ord[i][1]) < 0
                #    print(mat_el0,"   ", mat_el1, "\n")
                #    print("Corresponding eigenvalue is ", eigvals_ord[i],"\n")
                #    print("Absolute values of eigenvectors are: ", norm(vec0),norm(vec1),"\n .....................\n")
                #end
                #if abs(eigvals_ord[i][1])<8
                #    print("-----------------------\n Matrix Element 1: ", mat_el0, " Matrix Element 2: ", mat_el1, "\n")
                #    print("Eigenvalues", eigvals_ord[i],"\n ...................................\n")
                #end
                lambdas_conj[i] = -val2
            elseif mat_el1<0 && mat_el2>0
                lambdas[i] = val2
                if real(val2) < 0
                    print(mat_el1,"   ", mat_el2, "\n")
                end
                lambdas_conj[i] = -val1
                #if abs(eigvals_ord[i][1])<8
                #    print("-----------------------\n Matrix Element 1: ", mat_el0, " Matrix Element 2: ", mat_el1, "\n")
                #    print("Eigenvalues", eigvals_ord[i],"\n ...................................\n")
                #end
            else
                if abs(real(val1))<1e-10
                    imag_eigvals[i] = abs(imag(val1))
                    #print("Potential negative eigenvalue found. \n")
                    #print(eigvals_ord[i][1],",",eigvals_ord[i][2],"\n")
                else
                    Base.error("Error when evaluating the matrix elements!!!\n")
                end

            end 
        end
    end
    #print("Minimal lambda is ", minimum(lambdas),"\n")
    return lambdas, lambdas_conj, imag_eigvals
end

function get_gs_energy(lambdas,lambdas_conj, V, eps0)
    Lambda = Diagonal(vcat(lambdas,lambdas_conj))
    return  - 1/2 * sum(diag(V)) + 1/2 * sum(lambdas) #+ eps0#+ 1/4*sum(diag(Lambda))
end

PATH  = "C:/Users/Jan-Philipp/Documents/Eigene Dokumente/Physikstudium/6. Semester/Bachelorarbeit Quadratic Hamiltonians/N=40,lambda_IR=0.1+phi,lmabda_UV=10+phi/"
#PATH  = "C:/Users/Jan-Philipp/Documents/Eigene Dokumente/Physikstudium/6. Semester/Bachelorarbeit Quadratic Hamiltonians/N=200,lambda_IR=0.1,lambda_UV=10,phi=0.1/"
#PATH  = "C:/Users/Jan-Philipp/Documents/Eigene Dokumente/Physikstudium/6. Semester/Bachelorarbeit Quadratic Hamiltonians/N=xy,lambda_IR=0.1+phi,lmabda_UV=10+phi/"




gs_energies = []
imag_list = []
etas = []
imag_all = []
smallest_pos_eigvals = []
smllest_eigval = []
lambdas_list = []
labels = []
#p = plot()
for filename in readdir(PATH)
    eta_str = split(split(filename,'=')[2],',')[1]
    eta = parse(Float64,eta_str)
    print("eta = ",eta_str,"\n")
    if true
        filepath = PATH*filename
        #print(filepath*"\n")
        flat = npzread(filepath)  
        #print(length(flat),"\n")  
        om0,V0,W0,eps0 = unpack_arr(flat,N)     
        V0 = V0 + Diagonal(om0)
        vals, H = get_paired_eigenvalues(V0,W0)
        print("Found eigenvalues","\n")
        lambdas, lambdas_conj, imag_eigvals = classify_eigenvalues(vals,H)
        append!(gs_energies,[get_gs_energy(lambdas,lambdas_conj,V0,eps0)])
        append!(etas,[eta])
        append!(imag_list,[maximum(imag_eigvals)])
        append!(imag_all,imag_eigvals)
        append!(smallest_pos_eigvals,[minimum(lambdas[lambdas.>0])])
        append!(smllest_eigval,minimum(lambdas))
        append!(lambdas_list,[lambdas])
        append!(labels,[eta_str])
    end

    #if eta < -9
     #   print(lambdas)
    #end
    #if eta < -5
    #    plot!(p,collect(1:length(lambdas)),lambdas,seriestype=:scatter,label=eta_str)
    #end
end
#plot(p)
plot(etas,gs_energies,seriestype=:scatter)
#plot(etas,imag_list,seriestype=:scatter)
#plot!(etas,smallest_pos_eigvals,seriestype=:scatter)
#plot(etas,smllest_eigval,seriestype=:scatter)
#p = plot()

#for i in 1:length(etas)
#    if etas[i]<-7
#        plot!(p,collect(1:length(lambdas_list[i])),lambdas_list[i],seriestype=:scatter)
#    end
#end
#plot(p)




