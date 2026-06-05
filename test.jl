using ITensors
using ITensorMPS
using Printf

# load helpers
include("sites_mps.jl")
include("hamiltonian.jl")

function test_coeff_equal(N::Int, aspect::Float64; system::String="torus")
    maxdiff = 0.0
    argmax = nothing

    for k in 1:1
        for l in 0:1
            c_old = calc_fqh_coeff(k, l, N, aspect; system=system)
            c_new = calc_bkl_ferro(k, l, N, aspect; system=system)
            d = abs(c_old - c_new)

            if d > maxdiff
                maxdiff = d
                argmax = (k, l, c_old, c_new)
            end
        end
    end

    println("maxdiff = ", maxdiff)
    println("argmax  = ", argmax)
end

let
    N = 30
    aspect = 2.0
    test_coeff_equal(N, aspect)
end