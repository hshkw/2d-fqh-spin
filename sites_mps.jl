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
    # if conserved_nf 
    #     if conserved_sz
    #         if conserved_g
    #             return [
    #                 QN(("N", 0), ("Sz", 0), ("G", 0, modulus)) => 1,   # |empty⟩
    #                 QN(("N", 1), ("Sz", 1), ("G", pos, modulus)) => 1,   # |up⟩
    #                 QN(("N", 1), ("Sz", -1), ("G", pos, modulus)) => 1,   # |down⟩
    #                 QN(("N", 2), ("Sz", 0), ("G", 2pos, modulus)) => 1,   # |occupied⟩
    #             ]
    #         else
    #             return [
    #                 QN(("N", 0), ("Sz", 0)) => 1,   # |empty⟩
    #                 QN(("N", 1), ("Sz", 1)) => 1,   # |up⟩
    #                 QN(("N", 1), ("Sz", -1)) => 1,   # |down⟩
    #                 QN(("N", 2), ("Sz", 0)) => 1,   # |occupied⟩
    #             ]
    #         end
    #     else
    #         if conserved_g
    #             return [
    #                 QN(("N", 0), ("G", 0, modulus)) => 1,   # |empty⟩
    #                 QN(("N", 1), ("G", pos, modulus)) => 2,   # |up⟩, |down⟩
    #                 QN(("N", 2), ("G", 2pos, modulus)) => 1,   # |occupied⟩
    #             ]
    #         else
    #             return [
    #                 QN(("N", 0)) => 1,   # |empty⟩
    #                 QN(("N", 1)) => 2,   # |up⟩, |down⟩
    #                 QN(("N", 2)) => 1,   # |occupied⟩
    #             ]
    #         end
    #     end
    # else
    #     if conserved_sz
    #         if conserved_g
    #             return [
    #                 QN(("Sz", 0), ("G", 0, modulus)) => 1,   # |empty⟩
    #                 QN(("Sz", 1), ("G", pos, modulus)) => 1,   # |up⟩
    #                 QN(("Sz", -1), ("G", pos, modulus)) => 1,   # |down⟩
    #                 QN(("Sz", 0), ("G", 2pos, modulus)) => 1,   # |occupied⟩
    #             ]
    #         else
    #             return [
    #                 QN(("Sz", 0)) => 2,   # |empty⟩, |occupied⟩
    #                 QN(("Sz", 1)) => 1,   # |up⟩
    #                 QN(("Sz", -1)) => 1,   # |down⟩
    #             ]
    #         end
    #     else
    #         if conserved_g
    #             return [
    #                 QN(("G", 0, modulus)) => 1,   # |empty⟩
    #                 QN(("G", pos, modulus)) => 2,   # |up⟩, |down⟩
    #                 QN(("G", 2pos, modulus)) => 1,   # |occupied⟩ [TODO] needs particle num
    #             ]
    #         else
    #             return [QN() => 4]  # No conservation, just four states
    #         end
    #     end
    # end
    
    m = pos - 1
    if conserved_nf && conserved_sz && conserved_g
        return [
            QN(("N", 0), ("Sz", 0),  ("G", 0, modulus)) => 1,
            QN(("N", 1), ("Sz", 1),  ("G", m, modulus)) => 1,
            QN(("N", 1), ("Sz", -1), ("G", m, modulus)) => 1,
            QN(("N", 2), ("Sz", 0),  ("G", 2m, modulus)) => 1,
        ]
    elseif conserved_nf && conserved_sz
        return [
            QN(("N", 0), ("Sz", 0)) => 1,
            QN(("N", 1), ("Sz", 1)) => 1,
            QN(("N", 1), ("Sz", -1)) => 1,
            QN(("N", 2), ("Sz", 0)) => 1,
        ]
    elseif conserved_nf && conserved_g
        return [
            QN(("N", 0), ("G", 0, modulus)) => 1,
            QN(("N", 1), ("G", m, modulus)) => 2,
            QN(("N", 2), ("G", 2m, modulus)) => 1,
        ]
    elseif conserved_nf
        return [
            QN(("N", 0)) => 1,
            QN(("N", 1)) => 2,
            QN(("N", 2)) => 1,
        ]
    else
        return 4
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
function ITensors.op(::OpName"NupNdn", ::SiteType"FQHElectron", s::Index)
    o = emptyITensor(s', dag(s))
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
function ITensors.op(::OpName"N", ::SiteType"FQHElectron", s::Index)
    return op("Ntot", s)
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
function ITensors.op(::OpName"Sx", ::SiteType"FQHElectron", s::Index)
    o = emptyITensor(s', dag(s))
    o[s'=>2, s=>3] = 1.0
    o[s'=>3, s=>2] = 1.0
end
function ITensors.op(::OpName"Sy", ::SiteType"FQHElectron", s::Index)
    o = emptyITensor(s', dag(s))
    o[s'=>2, s=>3] = -1.0
    o[s'=>3, s=>2] = 1.0
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
function ITensors.op(::OpName"CupCdn", ::SiteType"FQHElectron", s::Index)
    o = emptyITensor(s', dag(s))
    # Cup Cdn |UpDn⟩ = - |Emp⟩
    o[s'=>1, s=>4] = -1.0
    return o
end

function ITensors.op(::OpName"CdagupCdagdn", ::SiteType"FQHElectron", s::Index)
    o = emptyITensor(s', dag(s))
    # Cdagup Cdagdn |Emp⟩ = |UpDn⟩
    o[s'=>4, s=>1] = 1.0
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
function build_fqh_nsg_sites(N::Int; conserved_nf::Bool=true,conserved_sz::Bool=true,  conserved_g::Bool=true, system::String="torus")::Vector{Index}
    modulus = (system == "torus") ? N : 1
    return [
        siteind(
            "FQHElectron"; 
            pos=j, 
            conserved_nf=conserved_nf, 
            conserved_sz=conserved_sz, 
            conserved_g=conserved_g, 
            modulus=modulus
        ) 
        for j in 1:N
    ]
end

# --初期MPS構築 -------------------------------------------------------------------
"""for initial state"""
function build_initial_state(N::Int; nu::Int=3)::Vector{Int}
    state = ones(Int, N)  # 1: empty, 2: Up
    for j in 2:nu:N
        state[j] = 2
    end
    flip = 1
    for j in (N-3flip+2):3:N
        state[j] = 3
    end
    return state
end
function build_initial_state2(N::Int; nu::Int=3)::Vector{Int}
    state = ones(Int, N)  # 1: empty, 3: Dn
    for j in 2:nu:N
        state[j] = 3
    end
    return state
end
function build_initial_state3(N::Int; nu::Int=3)::Vector{Int}
    state = ones(Int, N)  # 1: empty, 4: UpDn
    for j in nu-1:2nu:N
        state[j] = 2
    end
    for j in nu+2:2nu:N
        state[j] = 3
    end
    return state
end
function build_initial_state50(N::Int; nu::Int=3)::Vector{Int}
    state = ones(Int, N)  # 1: empty, 4: UpDn
    for j in nu-1:nu:120
        state[j] = 2
    end
    for j in 123:nu:N
        state[j] = 3
    end
    return state
end
function build_initial_state53(N::Int; nu::Int=3)::Vector{Int}
    # \nu = 5/3の非偏極状態
    state = ones(Int, N)  # 1: empty, 4: UpDn
    for j in 1:N
        state[j] = 4
    end
    frame = 2nu
    frrame = 0
    for j in 1+frame:2nu:60
        state[j] = 2
        state[j+nu] = 3
    end
    # state[61] = 1
    for j in 62:2nu:N-frame
        state[j] = 2
        state[j+nu] = 3
    end
    return state
end
function build_initial_state99(N::Int; nu::Int=3)::Vector{Int}
    state = ones(Int, N)  # 1: empty, 4: UpDn
    # for j in 2:6:N
    #     state[j] = 2
    #     state[j+3] = 3
    # end
    for j in 2:3:60
        state[j] = 2
    end
    # dn_pos = [
    #     20,23,
    #     41,44,47,
    #     62,65,68,71,
    #     83,86,89,92,95,
    #     104,107,110,113,116,119,
    # ]
    # for j in 61:3:N
    #     state[j] = 2
    # end
    # dn_pos_m = [
    #     20,23,
    #     41,44,47,
    #     61,64,67,70,
    #     82,85,88,91,94,
    #     103,106,109,112,115,118,
    # ]
    for j in 63:3:N
        state[j] = 2
    end
    dn_pos_p = [
        20,23,
        41,44,47,
        63,66,69,72,
        84,87,90,93,96,
        105,108,111,114,117,120,
    ]
    for dnp in dn_pos_p
        state[dnp] = 3
    end
    # for j in 2:3:60
    #     state[j] = 2
    # end
    # for j in 63:3:N
    #     state[j] = 3
    # end
    # state[59] = 3
    # # for j in 62:3:N
    # #     state[j] = 3
    # # end
    # ndn = 1
    # ndn_pos = N - 3 * ndn + 2
    # for j in N-1:-3:ndn_pos
    #     state[j] = 3
    # end
    # for j in 2:6:N
    #     state[j] = 2
    #     state[j+3] = 3
    # end
    return state
end
function build_initial_state96(N::Int; nu::Int=3)::Vector{Int}
    state = ones(Int, N)  # 1: empty, 4: UpDn
    for j in 2:3:48
        state[j] = 2
    end
    # dn_pos = [
    #     20,23,
    #     32,
    #     41,44,47,
    #     56,
    #     62,65,68,71,
    #     83,86,89,92,95,
    # ]
    # Ns - 1
    # for j in 49:3:N
    #     state[j] = 2
    # end
    # dn_pos_m = [
    #     20,23,
    #     32,
    #     41,44,47,
    #     55,
    #     61,64,67,70,
    #     82,85,88,91,94,
    # ]
    # for dnp in dn_pos_m
    #     state[dnp] = 3
    # end

    # Ns + 1
    for j in 51:3:N
        state[j] = 2
    end
    dn_pos_p = [
        20,23,
        32,
        41,44,47,
        57,
        63,66,69,72,
        84,87,90,93,96,
    ]
    for dnp in dn_pos_p
        state[dnp] = 3
    end
    return state
end
function build_initial_state72(N::Int; nu::Int=3)::Vector{Int}
    state = ones(Int, N)  # 1: empty, 4: UpDn
    for j in 2:3:36
        state[j] = 2
    end
    # dn_pos = [
    #     20,23,
    #     32,
    #     41,44,47,
    #     56,
    #     62,65,68,71,
    #     83,86,89,92,95,
    # ]
    # Ns - 1
    for j in 37:3:N
        state[j] = 2
    end
    dn_pos_m = [
        20,23,
        32,35,
        40,43,46,
        55,
        61,64,67,70,
    ]
    for dnp in dn_pos_m
        state[dnp] = 3
    end

    # Ns + 1
    # for j in 39:3:N
    #     state[j] = 2
    # end
    # dn_pos_p = [
    #     20,23,
    #     32,35,
    #     42,45,48,
    #     57,
    #     63,66,69,72,
    # ]
    # for dnp in dn_pos_p
    #     state[dnp] = 3
    # end
    return state
end
function build_initial_mps(N::Int, sites, init::Int=1)::MPS
    if init == 1
        state = build_initial_state(N)
    elseif init == 2
        state = build_initial_state2(N)
    elseif init == 3
        state = build_initial_state3(N)
    elseif init == 99
        state = build_initial_state99(N)
    elseif init == 96
        state = build_initial_state96(N)
    elseif init == 72
        state = build_initial_state72(N)
    elseif init == 50
        state = build_initial_state50(N)
    elseif init == 53
        state = build_initial_state53(N)
    else
        throw(ArgumentError("Invalid initial state index. Use 1, 2, or 3."))
    end
    # return random_mps(sites, state; linkdims=100)
    return MPS(sites, state)
end

