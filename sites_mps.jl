using ITensors
using ITensors.NDTensors
using ITensorMPS
# using KrylovKit: eigsolve
using Printf
using Dates
using Random

# ---------- サイト定義 -------------------------------------------------------------------
# 参考：https://github.com/ITensor/ITensorMPS.jl/blob/7ce812c42bfedcb3da1c250fdd5f19cb20394d4d/examples/src/electronk.jl#L98
function ITensors.space(::SiteType"FQHElectron";
    pos::Int=1, 
    conserved_nf::Bool=true, 
    conserved_sz::Bool=true, 
    conserved_g::Bool=true, 
    modulus::Int=1
)
    if conserved_nf 
        if conserved_sz
            if conserved_g
                return [
                    QN(("N", 0), ("Sz", 0), ("G", 0, modulus)) => 1,   # |empty⟩
                    QN(("N", 1), ("Sz", 1), ("G", pos, modulus)) => 1,   # |up⟩
                    QN(("N", 1), ("Sz", -1), ("G", pos, modulus)) => 1,   # |down⟩
                    QN(("N", 2), ("Sz", 0), ("G", 2pos, modulus)) => 1,   # |occupied⟩
                ]
            else
                return [
                    QN(("N", 0), ("Sz", 0)) => 1,   # |empty⟩
                    QN(("N", 1), ("Sz", 1)) => 1,   # |up⟩
                    QN(("N", 1), ("Sz", -1)) => 1,   # |down⟩
                    QN(("N", 2), ("Sz", 0)) => 1,   # |occupied⟩
                ]
            end
        else
            if conserved_g
                return [
                    QN(("N", 0), ("G", 0, modulus)) => 1,   # |empty⟩
                    QN(("N", 1), ("G", pos, modulus)) => 2,   # |up⟩, |down⟩
                    QN(("N", 2), ("G", 2pos, modulus)) => 1,   # |occupied⟩
                ]
            else
                return [
                    QN(("N", 0)) => 1,   # |empty⟩
                    QN(("N", 1)) => 2,   # |up⟩, |down⟩
                    QN(("N", 2)) => 1,   # |occupied⟩
                ]
            end
        end
    else
        if conserved_sz
            if conserved_g
                return [
                    QN(("Sz", 0), ("G", 0, modulus)) => 1,   # |empty⟩
                    QN(("Sz", 1), ("G", pos, modulus)) => 1,   # |up⟩
                    QN(("Sz", -1), ("G", pos, modulus)) => 1,   # |down⟩
                    QN(("Sz", 0), ("G", 2pos, modulus)) => 1,   # |occupied⟩
                ]
            else
                return [
                    QN(("Sz", 0)) => 1,   # |empty⟩
                    QN(("Sz", 1)) => 1,   # |up⟩
                    QN(("Sz", -1)) => 1,   # |down⟩
                    QN(("Sz", 0)) => 1,   # |occupied⟩
                ]
            end
        else
            if conserved_g
                return [
                    QN(("G", 0, modulus)) => 1,   # |empty⟩
                    QN(("G", pos, modulus)) => 2,   # |up⟩, |down⟩
                    QN(("G", 2pos, modulus)) => 1,   # |occupied⟩ [TODO] needs particle num
                ]
            else
                return [QN() => 4]  # No conservation, just four states
            end
        end
    end
end

ITensors.state(::StateName"Emp", ::SiteType"FQHElectron") = 1
ITensors.state(::StateName"Up", ::SiteType"FQHElectron") = 2
ITensors.state(::StateName"Dn", ::SiteType"FQHElectron") = 3
ITensors.state(::StateName"UpDn", ::SiteType"FQHElectron") = 4

ITensors.has_fermion_string(::OpName"Cup", ::SiteType"FQHElectron") = true
ITensors.has_fermion_string(::OpName"Cdagup", ::SiteType"FQHElectron") = true
ITensors.has_fermion_string(::OpName"Cdn", ::SiteType"FQHElectron") = true
ITensors.has_fermion_string(::OpName"Cdagdn", ::SiteType"FQHElectron") = true

function ITensors.op(::OpName"Nup", ::SiteType"FQHElectron", s::Index)
    o = emptyITensor(s', dag(s))
    o[s'=>2, s=>2] = 1.0  # |Up⟩⟨Up|
    o[s'=>4, s=>4] = 1.0  # |UpDn⟩⟨UpDn|
    return o
end
function ITensors.op(::OpName"Ndn", ::SiteType"FQHElectron", s::Index)
    o = emptyITensor(s', dag(s))
    o[s'=>3, s=>3] = 1.0  # |Dn⟩⟨Dn|
    o[s'=>4, s=>4] = 1.0  # |UpDn⟩⟨UpDn|
    return o
end
function ITensors.op(::OpName"Ntot", ::SiteType"FQHElectron", s::Index)
    o = emptyITensor(s', dag(s))
    o[s'=>2, s=>2] = 1.0  # |Up⟩⟨Up|
    o[s'=>3, s=>3] = 1.0  # |Dn⟩⟨Dn|
    o[s'=>4, s=>4] = 2.0  # |UpDn⟩⟨UpDn|
    return o
end

function ITensors.op(::OpName"Sz", ::SiteType"FQHElectron", s::Index)
    o = emptyITensor(s', dag(s))
    o[s'=>2, s=>2] = 1.0
    o[s'=>3, s=>3] = -1.0
    return o
end
function ITensors.op(::OpName"Splus", ::SiteType"FQHElectron", s::Index)
    o = emptyITensor(s', dag(s))
    o[s'=>2, s=>3] = 1.0  # |Up⟩⟨Dn|
    return o
end
function ITensors.op(::OpName"Sminus", ::SiteType"FQHElectron", s::Index)
    o = emptyITensor(s', dag(s))
    o[s'=>3, s=>2] = 1.0  # |Dn⟩⟨Up|
    return o
end

function ITensors.op(::OpName"Cdagup", ::SiteType"FQHElectron", s::Index)
    o = emptyITensor(s', dag(s))
    o[s'=>2, s=>1] = 1.0  # |Up⟩⟨empty|
    o[s'=>4, s=>3] = 1.0  # |UpDn⟩⟨Dn| UpDnの順なので符号は＋
    return o
end
function ITensors.op(::OpName"Cup", ::SiteType"FQHElectron", s::Index)
    o = emptyITensor(s', dag(s))
    o[s'=>1, s=>2] = 1.0  # |empty⟩⟨Up|
    o[s'=>3, s=>4] = 1.0  # |Dn⟩⟨UpDn|
    return o
end
function ITensors.op(::OpName"Cdagdn", ::SiteType"FQHElectron", s::Index)
    o = emptyITensor(s', dag(s))
    o[s'=>3, s=>1] = 1.0  # |Dn⟩⟨empty|
    o[s'=>4, s=>2] = -1.0  # |UpDn⟩⟨Up| UpDnの順なので符号は "マイナス"
    return o
end
function ITensors.op(::OpName"Cdn", ::SiteType"FQHElectron", s::Index)
    o = emptyITensor(s', dag(s))
    o[s'=>1, s=>3] = 1.0  # |empty⟩⟨Dn|
    o[s'=>2, s=>4] = -1.0  # |Up⟩⟨UpDn|
    return o
end

function ITensors.op(::OpName"F", ::SiteType"FQHElectron", s::Index)
    o = emptyITensor(s', dag(s))
    o[s'=>1, s=>1] = 1.0   # N=0
    o[s'=>2, s=>2] = -1.0  # N=1
    o[s'=>3, s=>3] = -1.0  # N=1
    o[s'=>4, s=>4] = 1.0   # N=2
    return o
end

function ITensors.op(::OpName"Proj0", ::SiteType"FQHElectron", s::Index)
    o = emptyITensor(s', dag(s))
    o[s'=>1, s=>1] = 1.0  # |empty⟩⟨empty|
    return o
end
function ITensors.op(::OpName"ProjUp", ::SiteType"FQHElectron", s::Index)
    o = emptyITensor(s', dag(s))
    o[s'=>2, s=>2] = 1.0  # |Up⟩⟨Up|
    return o
end
function ITensors.op(::OpName"ProjDn", ::SiteType"FQHElectron", s::Index)
    o = emptyITensor(s', dag(s))
    o[s'=>3, s=>3] = 1.0  # |Dn⟩⟨Dn|
    return o
end
function ITensors.op(::OpName"ProjUpDn", ::SiteType"FQHElectron", s::Index)
    o = emptyITensor(s', dag(s))
    o[s'=>4, s=>4] = 1.0  # |UpDn⟩⟨UpDn|
    return o
end

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

