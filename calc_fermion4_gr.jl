# Four-fermion correlations for LLL real-space pair correlation g(r;r1).
# Requires: ITensors, ITensorMPS, and ITensorCorrelators.jl
# Pkg.add(url="https://github.com/ITensor/ITensorCorrelators.jl.git")
#
# Modes: 1=(site 1, up), 2=(site 1, dn), 3=(site 2, up), ...
# Canonical pairs: a<b, c<d.
# Gamma[a,b;c,d] = <c†_a c†_b c_d c_c>
# The output stores only the Hermitian upper triangle (p<=q).
# Each pair belongs to the same (2Sz,G) sector, with G modulo N on torus.

using ITensors
using ITensorMPS
using ITensorCorrelators
using Printf

_f4_site(a::Int) = (a + 1) ÷ 2
_f4_spin(a::Int) = isodd(a) ? 1 : -1
_f4_spinname(a::Int) = isodd(a) ? "up" : "dn"
_f4_create(a::Int) = isodd(a) ? "Cdagup" : "Cdagdn"
_f4_destroy(a::Int) = isodd(a) ? "Cup" : "Cdn"

# The density channel has an independent spin-conserving assignment from
# the creation pair to the annihilation pair. For equal-spin pairs both
# possible assignments matter, so use the minimum transfer for filtering.
function _f4_transfer(a::Int, b::Int, c::Int, d::Int)::Int
    transfers = Int[]
    if _f4_spin(a) == _f4_spin(c) && _f4_spin(b) == _f4_spin(d)
        push!(transfers, abs(_f4_site(a) - _f4_site(c)))
    end
    if _f4_spin(a) == _f4_spin(d) && _f4_spin(b) == _f4_spin(c)
        push!(transfers, abs(_f4_site(a) - _f4_site(d)))
    end
    isempty(transfers) && error("Internal error: pair Sz is inconsistent")
    return minimum(transfers)
end

function _f4_mpo_element(psi::MPS, a::Int, b::Int, c::Int, d::Int)
    os = OpSum()
    os += (
        1.0,
        _f4_create(a), _f4_site(a),
        _f4_create(b), _f4_site(b),
        _f4_destroy(d), _f4_site(d),
        _f4_destroy(c), _f4_site(c),
    )
    W = MPO(os, siteinds(psi))
    return inner(psi', W, psi)
end

"""
    calc_fermion4_gr(psi, dir; system="edge", pair_sz=nothing,
                     anchor_site=nothing, anchor_radius=nothing,
                     max_transfer=nothing, batchsize=64,
                     backend=:correlator, dry_run=false)

Compute <c†_a c†_b c_d c_c>, restricted by conserved pair spin and momentum.

* `pair_sz` : one of -2, 0, +2 (ITensor integer 2Sz), or `nothing` for all.
* `anchor_site`, `anchor_radius` : keep pairs that each have at least one
  spin orbital within `anchor_site +/- anchor_radius` (approximation to g(r;r1)).
  Both must be supplied together, or both omitted.
* `max_transfer` : maximum absolute orbital momentum transfer |j1-j4|.
  For same-spin pairs keep if either direct/exchange transfer meets the cutoff.
  Omit for the exact full 2-RDM.
* `backend=:correlator` : batched MPS contractions with ITensorCorrelators.jl.
* `backend=:mpo` : slow, independent OpSum/MPO contraction (small-N testing).
* `dry_run` : print and return number of selected elements, without evaluating.

`fermion4_2rdm.csv` columns are:
a,b,c,d,i,si,j,sj,k,sk,l,sl,Sz_pair,G_pair,re,im
where i,si and j,sj label the creators a,b; k,sk and l,sl label c,d.
The annihilators appear in the order c_d c_c (i.e. l then k).
"""
function calc_fermion4_gr(
    psi::MPS,
    dir::String;
    system::String="edge",
    pair_sz::Union{Nothing,Int}=nothing,
    anchor_site::Union{Nothing,Int}=nothing,
    anchor_radius::Union{Nothing,Int}=nothing,
    max_transfer::Union{Nothing,Int}=nothing,
    batchsize::Int=64,
    backend::Symbol=:correlator,
    dry_run::Bool=false,
)
    system in ("edge", "torus") || error("system must be edge or torus")
    isnothing(pair_sz) || pair_sz in (-2, 0, 2) || error("pair_sz must be -2, 0, 2, or nothing")
    isnothing(anchor_site) == isnothing(anchor_radius) || error("Set anchor_site AND anchor_radius, or neither")
    isnothing(anchor_radius) || anchor_radius >= 0 || error("anchor_radius must be nonnegative")
    isnothing(max_transfer) || max_transfer >= 0 || error("max_transfer must be nonnegative")
    batchsize >= 1 || error("batchsize must be >= 1")
    backend in (:correlator, :mpo) || error("backend must be :correlator or :mpo")

    n = length(psi)
    if !isnothing(anchor_site)
        1 <= anchor_site <= n || error("anchor_site out of range")
    end

    Pair = NTuple{2,Int}
    groups = Dict{Tuple{Int,Int},Vector{Pair}}()
    for a in 1:(2*n-1), b in (a+1):(2*n)
        sz = _f4_spin(a) + _f4_spin(b)
        isnothing(pair_sz) || sz == pair_sz || continue
        if !isnothing(anchor_site)
            near_a = abs(_f4_site(a) - anchor_site) <= anchor_radius
            near_b = abs(_f4_site(b) - anchor_site) <= anchor_radius
            (near_a || near_b) || continue
        end
        g = _f4_site(a) + _f4_site(b) - 2
        system == "torus" && (g = mod(g, n))
        push!(get!(groups, (sz,g), Pair[]), (a,b))
    end

    # Do not store all quadruples in RAM: stream one sector at a time.
    function selected_quadruples(ps)
        return (
            (ps[p][1], ps[p][2], ps[q][1], ps[q][2])
            for p in eachindex(ps) for q in p:length(ps)
            if isnothing(max_transfer) || _f4_transfer(ps[p][1],ps[p][2],ps[q][1],ps[q][2]) <= max_transfer
        )
    end

    if dry_run
        counts = Dict{Int,Int}(-2=>0, 0=>0, 2=>0)
        for (szg, ps) in groups
            for _ in selected_quadruples(ps)
                counts[szg[1]] += 1
            end
        end
        @info "Selected 2-RDM elements" n counts total=sum(values(counts))
        return sum(values(counts))
    end

    mkpath(dir)
    outpath = joinpath(dir, "fermion4_2rdm.csv")
    @info "Starting 4-fermion expectation values" n system pair_sz anchor_site anchor_radius max_transfer backend outpath
    nwrite = 0

    # All elements in a correlator() batch must share operator names, the
    # site-index permutation, and the site-repetition pattern. The current
    # ITensorCorrelators.jl implementation uses only the first permutation.
    # The batch dictionary is flushed at least once per sector.
    function flush_batch!(io, key, chunk, sz, g)
        isempty(chunk) && return 0
        ops = key[1]
        coords = [
            (_f4_site(a), _f4_site(b), _f4_site(d), _f4_site(c))
            for (a,b,c,d) in chunk
        ]
        vals = if backend == :correlator
            result = correlator(psi, ops, coords)
            [result[x] for x in coords]
        else
            [_f4_mpo_element(psi,a,b,c,d) for (a,b,c,d) in chunk]
        end
        for ((a,b,c,d), value) in zip(chunk, vals)
            @printf(io, "%d,%d,%d,%d,%d,%s,%d,%s,%d,%s,%d,%s,%d,%d,%.16e,%.16e\n",
                a,b,c,d,
                _f4_site(a),_f4_spinname(a),
                _f4_site(b),_f4_spinname(b),
                _f4_site(c),_f4_spinname(c),
                _f4_site(d),_f4_spinname(d),
                sz,g,real(value),imag(value)
            )
        end
        return length(chunk)
    end

    open(outpath, "w") do io
        println(io, "a,b,c,d,i,si,j,sj,k,sk,l,sl,Sz_pair,G_pair,re,im")
        for (sz,g) in sort!(collect(keys(groups)))
            ps = groups[(sz,g)]
            buffers = Dict{Tuple{NTuple{4,String},NTuple{4,Int},NTuple{4,Int}},Vector{NTuple{4,Int}}}()
            for quad in selected_quadruples(ps)
                a,b,c,d = quad
                ops = (_f4_create(a), _f4_create(b), _f4_destroy(d), _f4_destroy(c))
                coords = (_f4_site(a), _f4_site(b), _f4_site(d), _f4_site(c))
                ordering = Tuple(sortperm(collect(coords)))
                repetitions = ntuple(k -> findfirst(==(coords[k]), coords)::Int, 4)
                key = (ops, ordering, repetitions)
                chunk = get!(buffers, key, NTuple{4,Int}[])
                push!(chunk, quad)
                if length(chunk) >= batchsize
                    nwrite += flush_batch!(io, key, chunk, sz, g)
                    empty!(chunk)
                end
            end
            for (key, chunk) in buffers
                nwrite += flush_batch!(io, key, chunk, sz, g)
            end
            @info "Finished sector" sz g nwrite
            flush(io)
        end
    end
    @info "Saved four-fermion 2-RDM" outpath nwrite
    return outpath
end

# Small smoke test. Include `sites_mps.jl` before calling this.
# `ITensorCorrelators.jl` must be validated against MPO for custom sites
# and repeated orbital positions before production calculations.
function test_fermion4_gr(; atol::Float64=1e-9)
    sites = build_fqh_nsg_sites(4;
        conserved_nf=true, conserved_sz=true, conserved_g=true, system="edge")
    psi = MPS(sites, [4, 1, 2, 3])
    # The pair (1up,1dn) and pair (3up,4dn) are occupied;
    # the pair (1up,2up) is not.
    for (a,b,c,d,expected) in (
        (1,2,1,2,1.0),
        (5,8,5,8,1.0),
        (1,8,1,8,1.0),
        (1,3,1,3,0.0),
        (2,5,2,5,1.0),
    )
        ops = (_f4_create(a),_f4_create(b),_f4_destroy(d),_f4_destroy(c))
        indices = (_f4_site(a),_f4_site(b),_f4_site(d),_f4_site(c))
        value_cor = correlator(psi, ops, [indices])[indices]
        value_mpo = _f4_mpo_element(psi,a,b,c,d)
        isapprox(value_cor,value_mpo; atol=atol, rtol=atol) ||
            error("Correlator/MPO disagree for $((a,b,c,d)): $value_cor vs $value_mpo")
        isapprox(value_cor,expected; atol=atol, rtol=atol) ||
            error("Wrong correlation for $((a,b,c,d)): $value_cor expected $expected")
    end
    @info "Four-fermion smoke test passed"
    return true
end
