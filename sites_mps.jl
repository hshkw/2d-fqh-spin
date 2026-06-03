using ITensors
using ITensors.NDTensors
using ITensorMPS
# using KrylovKit: eigsolve
using Printf
using Dates
using Random

# ---------- サイト定義 -------------------------------------------------------------------
function ITensors.space(::SiteType"FQHFermion"; pos::Int=1, conserved_nf::Bool=true, conserved_g::Bool=true, modulus::Int=1)
    if conserved_nf
        if conserved_g
            return [
                QN(("N", 0), ("G", 0, modulus)) => 1,   # |empty⟩
                QN(("N", 1), ("G", pos, modulus)) => 1,   # |occupied⟩
            ]
        else
            return [
                QN(("N", 0)) => 1,   # |empty⟩
                QN(("N", 1)) => 1,   # |occupied⟩
            ]
        end
    else
        return [QN() => 1, QN() => 1]  # No conservation, just two states
    end
end
ITensors.has_fermion_string(::OpName"C", ::SiteType"FQHFermion") = true
ITensors.has_fermion_string(::OpName"Cdag", ::SiteType"FQHFermion") = true
function ITensors.op(::OpName"N", ::SiteType"FQHFermion", s::Index)
    return op("Proj1", s)  # |1⟩⟨1| (occupied projector)
end
function ITensors.op(::OpName"Cdag", ::SiteType"FQHFermion", s::Index)
    o = emptyITensor(s', dag(s))
    o[s'=>2, s=>1] = 1.0  # |occupied⟩⟨empty|
    return o
end
function ITensors.op(::OpName"C", ::SiteType"FQHFermion", s::Index)
    o = emptyITensor(s', dag(s))
    o[s'=>1, s=>2] = 1.0  # |empty⟩⟨occupied|
    return o
end
function ITensors.op(::OpName"F", ::SiteType"FQHFermion", s::Index)
    o = emptyITensor(s', dag(s))
    o[s'=>1, s=>1] = 1.0
    o[s'=>2, s=>2] = -1.0
    return o
end
function ITensors.op(::OpName"Ntot", ::SiteType"FQHFermion", s::Index)
    return op("N", s)
end
function ITensors.op(::OpName"Proj0", ::SiteType"FQHFermion", s::Index)
    o = emptyITensor(s', dag(s))
    o[s'=>1, s=>1] = 1.0  # |empty⟩⟨empty|
    return o
end
function ITensors.op(::OpName"Proj1", ::SiteType"FQHFermion", s::Index)
    o = emptyITensor(s', dag(s))
    o[s'=>2, s=>2] = 1.0  # |occupied⟩⟨occupied|
    return o
end
ITensors.state(::StateName"Emp", ::SiteType"FQHFermion") = 1
ITensors.state(::StateName"Occ", ::SiteType"FQHFermion") = 2

"""
for site index
"""
function build_fqh_sites(N::Int; conserved_nf::Bool=true, conserved_g::Bool=true, system::String="torus")::Vector{Index}
    modulus = (system == "torus") ? N : 1
    return [siteind("FQHFermion"; pos=j, conserved_nf=conserved_nf, conserved_g=conserved_g, modulus=modulus) for j in 1:N]
end

# --初期MPS構築 -------------------------------------------------------------------
"""for initial state
"""
function build_initial_state(N::Int)::Vector{Int}
    state = ones(Int, N)  # 1: empty, 2: occupied
    for j in 2:3:N
        state[j] = 2
    end
    return state
end
function build_initial_state2(N::Int)::Vector{Int}
    state = ones(Int, N)  # 1: empty, 2: occupied
    for j in 1:3:N
        state[j] = 2
    end
    return state
end
function build_initial_state3(N::Int)::Vector{Int}
    state = ones(Int, N)  # 1: empty, 2: occupied
    for j in 3:3:N
        state[j] = 2
    end
    return state
end
function build_initial_mps(N::Int, sites, init::Int=1)::MPS
    if init == 1
        state = build_initial_state(N)
    elseif init == 2
        state = build_initial_state2(N)
    elseif init == 3
        state = build_initial_state3(N)
    else
        throw(ArgumentError("Invalid initial state index. Use 1, 2, or 3."))
    end
    return MPS(sites, state)
end

