#Path of this file: "C:/Users/Jan-Philipp/Documents/Eigene Dokumente/Physikstudium/6. Semester/Bachelor-Thesis/Numerics/without_n_dep/Julia/with_offset_phi/Bogoliubov_Transformation.jl"
import LinearAlgebra, NPZ, Base, Plots, DelimitedFiles
using LinearAlgebra, NPZ, Plots, DelimitedFiles

N = 200

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
    eigs = eigen(H)
    eigvals = eigs.values
    eigvecs = Transpose(eigs.vectors)
    #print(det(eigvecs),"\n")
    sortind = sortperm(Float64.(real(eigvals)))
    sorted_vals = eigvals[sortind]
    sorted_vecs = eigvecs[sortind,:]
    eigvals_ord = collect(zip(sorted_vals[1:Int(length(eigvals)/2)],(sorted_vals[Int(length(eigvals)/2)+1:length(eigvals)])[end:-1:1]))
    eigvecs_ord = [[sorted_vecs[i,:], sorted_vecs[length(eigvals)-i+1,:]] for i = 1:length(eigvals)÷2]   
    for i in 1:length(eigvals_ord)
        if norm(H*eigvecs_ord[i][1]-eigvals_ord[i][1]*eigvecs_ord[i][1])>1e-10 || norm(H*eigvecs_ord[i][2]-eigvals_ord[i][2]*eigvecs_ord[i][2])>1e-10
            Base.error("Error when ordering the eigenvalues\n")
        end
    end
    return eigvals_ord, eigvecs_ord
end

function classify_eigenvalues(eigvals_ord, eigvecs_ord)
    imag_eigvals = zeros(length(eigvals_ord))
    lambdas = zeros(length(eigvals_ord))
    lambdas_conj = zeros(length(eigvals_ord))
    for (i,vecs) in enumerate(eigvecs_ord)
        vec0 = vecs[1]
        vec1 = vecs[2]
        mat_el0 = Float64(real(conj(Transpose(vec0))*Omega*vec0))
        mat_el1 = Float64(real(conj(Transpose(vec1))*Omega*vec1))
        #if abs(mat_el0)<0.9
        #    print("Matrix Element 1: ", mat_el0, " Matrix Element 2: ", mat_el1, "\n")
        #    print("Eigenvalues", eigvals_ord[i],"\n ...................................\n")
        #end
        if mat_el0>0 && mat_el1<0
            lambdas[i] = eigvals_ord[i][1]
            if abs(imag(eigvals_ord[i][1])) > 1e-2
                print(mat_el0,"   ", mat_el1, "\n")
            end
            #if real(eigvals_ord[i][1]) < 0
            #    print(mat_el0,"   ", mat_el1, "\n")
            #    print("Corresponding eigenvalue is ", eigvals_ord[i],"\n")
            #    print("Absolute values of eigenvectors are: ", norm(vec0),norm(vec1),"\n .....................\n")
            #end
            #if abs(eigvals_ord[i][1])<8
            #    print("-----------------------\n Matrix Element 1: ", mat_el0, " Matrix Element 2: ", mat_el1, "\n")
            #    print("Eigenvalues", eigvals_ord[i],"\n ...................................\n")
            #end
            lambdas_conj[i] = -eigvals_ord[i][2]
        elseif mat_el0<0 && mat_el1>0
            lambdas[i] = eigvals_ord[i][2]
            if abs(imag(eigvals_ord[i][2])) > 1e-2
                print(mat_el0,"   ", mat_el1, "\n")
            end
            lambdas_conj[i] = -eigvals_ord[i][1]
            #if abs(eigvals_ord[i][1])<8
            #    print("-----------------------\n Matrix Element 1: ", mat_el0, " Matrix Element 2: ", mat_el1, "\n")
            #    print("Eigenvalues", eigvals_ord[i],"\n ...................................\n")
            #end
        else
            if abs(real(eigvals_ord[i][1]))<1e-10
                imag_eigvals[i] = abs(imag(eigvals_ord[i][1]))
                print("-----------------------\n Matrix Element 1: ", mat_el0, " Matrix Element 2: ", mat_el1, "\n")
                #print("Potential negative eigenvalue found. \n")
                #print(eigvals_ord[i][1],",",eigvals_ord[i][2],"\n")
            else
                Base.error("Error when evaluating the matrix elements!!!\n")
            end

        end 
    end
    #print("Minimal lambda is ", minimum(lambdas),"\n")
    return lambdas, lambdas_conj, imag_eigvals
end

function get_gs_energy_diff(lambdas,lambdas_conj, V, eps0)
    Lambda = Diagonal(vcat(lambdas,lambdas_conj))
    return  - 1/2 * sum(diag(V)) + 1/2 * sum(lambdas) #+ eps0#+ 1/4*sum(diag(Lambda))
end

#PATH  = "C:/Users/Jan-Philipp/Documents/Eigene Dokumente/Physikstudium/6. Semester/Bachelorarbeit Quadratic Hamiltonians/N=40,lambda_IR=0.1+phi,lmabda_UV=10+phi/"
#PATH  = "C:/Users/Jan-Philipp/Documents/Eigene Dokumente/Physikstudium/6. Semester/Bachelorarbeit Quadratic Hamiltonians/N=200,lambda_IR=0.1,lambda_UV=10,phi=0.1/"
#PATH  = "C:/Users/Jan-Philipp/Documents/Eigene Dokumente/Physikstudium/6. Semester/Bachelorarbeit Quadratic Hamiltonians/N=xy,lambda_IR=0.1+phi,lmabda_UV=10+phi/"
PATH  =  "C:/Users/Jan-Philipp/Documents/Eigene Dokumente/Physikstudium/6. Semester/Bachelorarbeit Quadratic Hamiltonians/N=200,lambda_IR=0.1,lambda_UV=10.0/"




gs_energies_diff = []
imag_list = []
etas = []
imag_all = []
smallest_pos_eigvals = []
smllest_eigval = []
lambdas_list = []
lambdas_conj_list = []
labels = []
epss0 = []
#p = plot()
for filename in readdir(PATH)
    eta_str = split(split(filename,'=')[2],',')[1]
    eta = parse(Float64,eta_str)
    print("eta = ",eta_str,"\n")
    filepath = PATH*filename
    #print(filepath*"\n")
    flat = npzread(filepath)  
    #print(length(flat),"\n")  
    om0,V0,W0,eps0 = unpack_arr(flat,N)     
    V0 = V0 + Diagonal(om0)
    W0 = 1/2 * (W0 + W0')
    vals,vecs = get_paired_eigenvalues(V0,W0)
    lambdas, lambdas_conj, imag_eigvals = classify_eigenvalues(vals,vecs)
    #print(vals,"\n")
    append!(gs_energies_diff,[get_gs_energy_diff(lambdas,lambdas_conj,V0,eps0)])
    append!(etas,[eta])
    append!(imag_list,[maximum(imag_eigvals)])
    append!(imag_all,imag_eigvals)
    append!(smallest_pos_eigvals,[minimum(lambdas[lambdas.>0])])
    append!(smllest_eigval,minimum(lambdas))
    append!(lambdas_list,[lambdas])
    append!(lambdas_conj_list,[lambdas_conj])
    append!(labels,[eta_str])
    append!(epss0,[eps0])

    #if eta < -9
     #   print(lambdas)
    #end
    #if eta < -5
    #    plot!(p,collect(1:length(lambdas)),lambdas,seriestype=:scatter,label=eta_str)
    #end
end
#plot(p)
#plot(etas,gs_energies,seriestype=:scatter)
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
out = [etas,lambdas_list,imag_all,N]#float.(out)#float.(vcat(float.(etas), float.(lambdas_list), float.(imag_all), N))

SAVEPATH_base = "C:/Users/Jan-Philipp/Documents/Eigene Dokumente/Physikstudium/6. Semester/Bachelor-Thesis/Numerics/without_n_dep/Julia/with_offset_phi/outfiles/Bogoliubov,N=200,phi=0.1_.csv"


writedlm(SAVEPATH, out)



