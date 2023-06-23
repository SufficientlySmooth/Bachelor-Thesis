#Path of this file: "C:/Users/Jan-Philipp/Documents/Eigene Dokumente/Physikstudium/6. Semester/Bachelor-Thesis/Numerics/without_n_dep/Julia/with_offset_phi/Bogoliubov_without_displacement.jl"
import LinearAlgebra, NPZ, Base, Plots
using LinearAlgebra, NPZ, Plots

N = 50 
Omega = [Diagonal(ones(N)) zeros(N,N); zeros((N,N)) -Diagonal(ones(N))]
theta = [zeros((N,N)) Diagonal(ones(N)); Diagonal(ones(N)) zeros((N,N))] #missing complex conjugation operator!

function om(k,consts)
    xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L,phi = consts
    return  cc*abs(k)*sqrt(1+(k^2*xi^2)/2)
end

function W(k,consts)
    xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L,phi = consts
    return (k^2*xi^2/(2+k^2*xi^2))^(1/4)
end

function c(k,consts)
    xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L,phi = consts
    w = W(k,consts)
    return 1/2*(w+1/w)
end

function s(k,consts)
    xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L,phi = consts
    w = W(k,consts)
    return 1/2*(1/w-w)
end

function gen_1Dgrid(consts)
    xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L,phi = consts
    n_min = Int(ceil(lamb_IR/dk))
    n_max = Int(floor(lamb_UV/dk))
    k_pos = collect(n_min:n_max)*dk 
    k_pos += phi/L*ones(length(k_pos))
    k_neg = -k_pos[end:-1:1] 
    k_neg += phi/L*ones(length(k_neg))
    return vcat(k_neg,k_pos)
end

function V0(k,k_,consts)
    xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L,phi = consts
    return g_ib/(2*pi)*dk*(c(k,consts)*c(k_,consts)+s(k,consts)*s(k_,consts))
end

function W0(k,k_,consts)
    xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L,phi = consts
    return g_ib*dk/(2*pi)*s(k,consts)*c(k_,consts)*(-1)
end

function eps0(grid,consts)
    xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L,phi = consts
    g_ib*n0+g_ib/(2*pi)*dk*sum([s(k,consts)^2 for k in grid])
end

function omega0(k,consts)
    xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L,phi = consts
    return om(k,consts)
end

function W0_tilde(k,consts)
    xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L,phi = consts
    return  g_ib*sqrt(n0)/sqrt(2*pi)*sqrt(dk)*W(k,consts)
end

function omega0_arr(grid,consts)
    xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L,phi = consts
    return [omega0(k,consts) for k in grid]
end

function W0_tilde_arr(grid,consts)
    xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L,phi = consts
    return [W0_tilde(k,consts) for k in grid]
end

function W0_arr(grid,consts)
    xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L,phi = consts
    N = length(grid)
    return reshape([W0(k,k_,consts) for k_ in grid for k in grid],(N,N))
end

function V0_arr(grid,consts)
    xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L,phi = consts
    N = length(grid)
    return reshape([V0(k,k_,consts) for k_ in grid for k in grid],(N,N))
end


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
    sortind = sortperm(Float64.(real(eigvals)))
    sorted_vals = eigvals[sortind]
    sorted_vecs = eigvecs[sortind,:]
    max = 0
    maxind = [1,1]
    for (i,val1) in enumerate(sorted_vals)
        for (j,val2) in enumerate(sorted_vals)
            vec1 = sorted_vecs[i]
            vec2 = sorted_vecs[j]
            if vec1!=vec2 && abs(vec1'*vec2)>max
                max = abs(vec1'*vec2)
                maxind[1] = i
                maxind[2] = j
            end
        end
    end
    print("Maximal overlap is ",max,"\n")
    print("The eigenvectors correspondend to the eigenvalues ",sorted_vals[maxind[1]]," and ",sorted_vals[maxind[2]],"\n")
    print("Transformation matrix determinant is ",det(eigvecs),"\n")
    eigvals_ord = collect(zip(sorted_vals[1:Int(length(eigvals)/2)],(sorted_vals[Int(length(eigvals)/2)+1:length(eigvals)])[end:-1:1]))
    eigvecs_ord = [[sorted_vecs[i,:], sorted_vecs[length(eigvals)-i+1,:]] for i = 1:length(eigvals)÷2]   
    for i in 1:length(eigvals_ord)
        if norm(H*eigvecs_ord[i][1]-eigvals_ord[i][1]*eigvecs_ord[i][1])>1e-10 || norm(H*eigvecs_ord[i][2]-eigvals_ord[i][2]*eigvecs_ord[i][2])>1e-10
            Base.error("Error when ordering the eigenvalues\n")
        end
    end
    return eigvals_ord, eigvecs_ord, H
end

function classify_eigenvalues(eigvals_ord, eigvecs_ord,H)
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
        if mat_el0>0 && mat_el1<0 && abs(imag(eigvals_ord[i][1]))<1e-10 && abs(imag(eigvals_ord[i][2]))<1e-10
            lambdas[i] = eigvals_ord[i][1]
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
        elseif mat_el0<0 && mat_el1>0 && abs(imag(eigvals_ord[i][1]))<1e-10 && abs(imag(eigvals_ord[i][2]))<1e-10
            lambdas[i] = eigvals_ord[i][2]
            if real(eigvals_ord[i][2]) < 0
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
                #=
                print("--------------------------------------------\n")
                print(eigvals_ord[i],"\n")
                #print(norm(vec0-vec1),"\n")
                #print(norm(vec0-conj(vec1)),"\n")
                a0 = vec(nullspace((H-diagm(ones(2*N)*(-conj(eigvals_ord[i][1]))))))
                a1 = vec(nullspace((H-diagm(ones(2*N)*(-conj(eigvals_ord[i][2]))))))
                #atest = vec(nullspace((H-diagm(ones(2*N)*(pi)))))
                print(norm(H*a0-(-conj(eigvals_ord[i][1]))*a0),"\n")
                print(norm(H*a1-(-conj(eigvals_ord[i][2]))*a1),"\n")
                #print(norm(H*vec0-eigvals_ord[i][1]*vec0),"\n")
                print("(",-conj(eigvals_ord[i][1]),") and (",-conj(eigvals_ord[i][2]),") are also good eigenvalues!!!\n")
                #print(norm(H*atest-(pi)*atest),"\n")
                
                if length(a1)==2*N
                    print(norm(a0/norm(a0)-conj(vec0)),"\n")
                    print(norm(a0/norm(a0)-conj(vec1)),"\n")
                    print(norm(a0/norm(a0)+conj(vec0)),"\n")
                    print(norm(a0/norm(a0)+conj(vec1)),"\n")
                    print(norm(a0/norm(a0)-(vec0)),"\n")
                    print(norm(a0/norm(a0)-(vec1)),"\n")
                    print(norm(a0/norm(a0)+(vec0)),"\n")
                    print(norm(a0/norm(a0)+(vec1)),"\n")
                end
                
                #print(".........\n")
                #print(eigvals_ord,"\n")
                print("--------------------------------------------\n")
                =#

                #print(imag_eigvals[i],"\n")
                #a = vec(nullspace((H-diag(ones(2*N)*(eigvals_ord[i][1])))))
                #print(length(a),"\n ------------\n")
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

function get_gs_energy(lambdas,lambdas_conj, V, eps0)
    Lambda = Diagonal(vcat(lambdas,lambdas_conj))
    return  - 1/2 * sum(diag(V)) + 1/2 * sum(lambdas) + eps0#+ 1/4*sum(diag(Lambda))
end


etas = collect(LinRange(-20,20,101))
gs_energies = []
imag_list = []
imag_all = []
smallest_pos_eigvals = []
smllest_eigval = []
lambdas_list = []

for eta in etas
    print("eta = ", eta, "\n")
    xi = 1 #characteristic length (BEC healing length)
    cc = 1 #c in free Bogoliubov dispersion (speed of sound)
    lamb_IR = 1e-1 #Infrared cutoff
    lamb_UV = 1e1 #ultra-violet cutoff
    m_b = 1/(sqrt(2)*cc*xi) #reduced mass = boson mass in the limit of infinite impurity mass
    #eta = 1 #will be varied between -10...10 later
    phi = 1e-1 #offset phi/L for k values
    n0 = 1.05/xi #
    gamma = 0.438
    g_bb = gamma*n0/m_b
    a_bb = -2/(m_b*g_bb)
    g_ib = eta*g_bb
    dk = 4e-1
    L = 2*pi/dk
    consts = (xi,cc,lamb_IR,lamb_UV,m_b,eta,n0,gamma,a_bb,g_bb,g_ib,dk,L,phi)

    grid = gen_1Dgrid(consts)

    om0, V0, W0, eps0 = (omega0_arr(grid,consts),V0_arr(grid,consts),W0_arr(grid,consts),0)
    V0 = V0 + Diagonal(om0)
    W0 = 1/2*(W0+W0')
    vals,vecs,H = get_paired_eigenvalues(V0,W0)
    lambdas, lambdas_conj, imag_eigvals = classify_eigenvalues(vals,vecs,H)
    append!(gs_energies,[get_gs_energy(lambdas,lambdas_conj,V0,eps0)])
    append!(imag_list,[maximum(imag_eigvals)])
    append!(imag_all,imag_eigvals)
    append!(smallest_pos_eigvals,[minimum(lambdas[lambdas.>0])])
    append!(smllest_eigval,minimum(lambdas))
    append!(lambdas_list,[lambdas])
    
end

plot(etas,gs_energies)

#plot(etas,imag_list,seriestype=:scatter)
#plot(etas,smallest_pos_eigvals)
#plot(etas,smllest_eigval)


