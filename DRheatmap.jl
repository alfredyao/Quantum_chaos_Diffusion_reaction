using Random, Distributions, Plots, Statistics, SpecialFunctions, Base.Threads, QuadGK, Interpolations
using Roots

N1 = 10^3
Nx = 120
pl = 0.1
pr = 0.1
lam = 0.2
sam = 10^4
spaceing=5
nit_values = spaceing:spaceing:120 # Range of nit values

function mypostive(x::Real)
    if x >= 0
        return x
    else
        return 0
    end
end

function sampleMN(x)
    return rand(Multinomial(x,[pl,pr,lam,mypostive(lam*(x-1)/N1),1-pl-pr-lam-mypostive(lam*(x-1)/N1)]))
end

nmeans = zeros(length(nit_values), Nx)


@threads for nit_idx in 1:length(nit_values)
    nit = nit_values[nit_idx]
    n = zeros(Int64, sam, Nx)
    
    for c in 1:sam
        n[c, round(Int, Nx/2-0):round(Int, Nx/2+0)] .= 1

        for k in 1:nit
            nup = zeros(5, Nx)
            for i in 1:Nx
                nup[:, i] = sampleMN(n[c, i])
            end
            n[c,2:Nx-1]=   .+ nup[3,2:Nx-1] .- nup[4,2:Nx-1] .+ nup[1,3:Nx] .+ nup[2,1:Nx-2] .+ n[c,2:Nx-1] #for the normal DR process add these two .- nup[1,2:Nx-1] .- nup[2,2:Nx-1] 
            n[c,1]=nup[3,1]-nup[4,1]+nup[1,2]+nup[2,Nx]+n[c,1] #left boundary  # add - nup[1,1] - nup[2,1] for normal DR
            n[c,Nx]=nup[3,Nx]-nup[4,Nx]+nup[1,1]+nup[2,Nx-1]+n[c,Nx] #right boundary  # add - nup[1,Nx] - nup[2,Nx] for normal DR
        end
    end

    nmean = mapslices(mean, n, dims=1)
    nmeans[nit_idx, :] = transpose(nmean)
end

using Interpolations, Plots



# Create a finer grid for interpolation
x_range = 1:size(nmeans, 2)
y_range = 1:size(nmeans, 1)

# Interpolation object
interp = interpolate((y_range, x_range), nmeans, Gridded(Linear()))

# Create a finer grid
fine_x = range(1, stop=size(nmeans, 2), length=500)
fine_y = range(1, stop=size(nmeans, 1), length=500)
nmeans_fine = [interp(y, x) for y in fine_y, x in fine_x]

# Create the heatmap with the fine grid
fig1=heatmap(fine_x, fine_y*spaceing, nmeans_fine, xlabel="x", ylabel="n", colorbar=false, title="Heatmap of otoc")

# Save the heatmap to a file
savefig("/Users/shunyuyao/Desktop/nmeans_heatmap.png")

# Display the heatmap
display(heatmap(fine_x, fine_y*spaceing, nmeans_fine, xlabel="x", ylabel="n", colorbar=false, title="Heatmap of otoc"))