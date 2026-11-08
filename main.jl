using ITensors
using ITensorMPS
using Printf
using Dates
using Random

include("sites_mps.jl")
include("hamiltonian.jl")



# ---DMRG オブザーバー-----------------------------------------------------------------------
mutable struct MemoryDMRGObserver
    inner::DMRGObserver
    sweeps_path::String
    write_path::String  # DiskProjMPO が LR を書き出すディレクトリ
end

function MemoryDMRGObserver(eps::Float64; dir::String, minsweeps::Int=1, write_path::String="tmpout")
    sweep_path_ = joinpath(dir, "sweeps.csv")
    open(sweep_path_, "w") do io
        header = "sweep,energy,truncation_error,maxlinkdim,diff"
        println(io, header)
    end

    return MemoryDMRGObserver(DMRGObserver(; energy_tol=eps, minsweeps=minsweeps), sweep_path_, write_path)
end
function _lr_disk_bytes(write_path::String)::Int
    isdir(write_path) || return 0
    total = 0
    for (root, _, files) in walkdir(write_path)
        for f in files
            total += filesize(joinpath(root, f))
        end
    end
    return total
end
function ITensorMPS.checkdone!(o::MemoryDMRGObserver; kwargs...)
    cur_eg = real(energies(o.inner))[end]
    prev_eg = length(energies(o.inner)) > 1 ? real(energies(o.inner))[end-1] : 0.0
    diff_eg = abs(cur_eg - prev_eg)
    max_trcerr = truncerrors(o.inner)[end]
    sweep = kwargs[:sweep]
    psi = kwargs[:psi]
    maxdim = maxlinkdim(psi)
    @info @sprintf(
        "   energy=%s diff=%s  maxlinkdim=%d maxerr=%.2E\n",
        cur_eg, diff_eg, maxdim, max_trcerr
    )
    open(o.sweeps_path, "a") do io
        line = "$(sweep),$(cur_eg)"
        line *= ",$(max_trcerr)"
        line *= ",$(maxdim)"
        line *= ",$(diff_eg)"
        println(io, line)
    end
    if (
        length(real(energies(o.inner))) > o.inner.minsweeps &&
        abs(real(energies(o.inner))[end] - real(energies(o.inner))[end-1]) < o.inner.etol
    )
        @info @sprintf("Energy difference less than %.1E, stopping DMRG", o.inner.etol)
        return true
    end

    return false
end
function ITensorMPS.measure!(o::MemoryDMRGObserver; kwargs...)
    measure!(o.inner; kwargs...)

    sweep_is_done = kwargs[:sweep_is_done]
    sweep = kwargs[:sweep]
    psi = kwargs[:psi]
    projected_operator = kwargs[:projected_operator]
    bond = kwargs[:bond]
    half_sweep = kwargs[:half_sweep]
    peak_mem = Sys.maxrss() / 1024^3

    # 中央ボンド (ha=1, →方向) で計測:
    #   ProjMPO     → LR 全体がメモリ上にあり summarysize が正確
    #   DiskProjMPO → Lcache (L[mid-1]) と Rcache (R[mid+2]) が最大サイズ;
    #                 ディスク上の LR はファイルサイズで別途取得
    # MPO リンク次元 W:
    #   W_left   = L環境テンソルと H[bond] の間のリンク次元
    #   W_center = H[bond] と H[bond+1] の間のリンク次元
    #   W_right  = H[bond+1] と R環境テンソルの間のリンク次元
    N = length(psi)
    mid = N ÷ 2
    if bond == mid && half_sweep == 1
        H = projected_operator.H
        N_H = length(H)
        W_left = bond >= 2 ? dim(linkind(H, bond - 1)) : 1
        W_center = dim(linkind(H, bond))
        W_right = bond + 1 < N_H ? dim(linkind(H, bond + 1)) : 1

        psi_size = Base.format_bytes(Base.summarysize(psi))
        ph_mem = Base.summarysize(projected_operator)
        disk_lr = _lr_disk_bytes(o.write_path)
        ph_mem_str = Base.format_bytes(ph_mem)
        ph_total_str = Base.format_bytes(ph_mem + disk_lr)
        disk_active = isdir(o.write_path) && !isempty(readdir(o.write_path))
        @info @sprintf(
            "Sweep(ha=1) %d, bond %d (center), W=(%d,%d,%d), |psi|=%s, |PH|:in-mem=%s, disk=%s, disk+mem=%s, disk_active=%s, RSS=%.2f GB",
            sweep, bond, W_left, W_center, W_right,
            psi_size, ph_mem_str, Base.format_bytes(disk_lr), ph_total_str,
            string(disk_active), peak_mem
        )
    end
end
# ----スイープスケジュール------------------------------------------------
function make_sweeps(maxdim::Int)::Sweeps
    sweep_num = 50
    dims = [20]
    wnum = 1
    while last(dims) * 2 < maxdim
        push!(dims, last(dims) * 2)
        wnum += 1
    end
    for _ in 1:sweep_num
        push!(dims, maxdim)
    end

    noises = [(
        if i <= wnum
            1e-6
        else
            0.0
        end
    ) for i in 1:length(dims)]

    nsweeps = length(dims)
    sweeps = Sweeps(nsweeps)
    maxdim!(sweeps, dims...)
    noise!(sweeps, noises...)

    # maxdim_schedule = [200, 200, 200, 200, 200, 200, 200]
    # mindim_schedule = [200, 200, 200, 200, 200, 200, 200] # Forces dimension size
    # cutoff_schedule = [1e-8, 1e-8, 1e-8, 1e-7, 1e-6, 1e-6, 1e-6] # Your desired loose cutoff
    # noise_schedule = [1e-6, 1e-7, 1e-8, 1e-8, 0, 0, 0]
    # nsweeps = length(maxdim_schedule)
    # sweeps = Sweeps(nsweeps)
    # maxdim!(sweeps, maxdim_schedule...)
    # mindim!(sweeps, mindim_schedule...)
    # cutoff!(sweeps, cutoff_schedule...)
    # noise!(sweeps, noise_schedule...)

    return sweeps
end
# ---期待値計算-----------------------------------------------------------------------
function calc_orbital_occupation_expectations(psi::MPS, dir::String; file::String = "orb_occ_exp.csv")
    mkpath(dir)

    output_path = joinpath(dir, file)

    nj_exp = expect(psi, "N")
    nupj_exp = expect(psi, "Nup")
    ndnj_exp = expect(psi, "Ndn")

    N = length(psi)

    open(output_path, "w") do io
        println(io, "j,ndn,nup,ntot")

        for j in 1:N
            @printf(
                io, "%d,%.10f,%.10f,%.10f\n",
                j, ndnj_exp[j], nupj_exp[j], nj_exp[j],
            )
        end

        @printf(
            io, "%d,%.10f,%.10f,%.10f\n",
            -1, sum(ndnj_exp), sum(nupj_exp),sum(nj_exp),
        )
    end

    return nothing
end
function calc_nj_expectation(psi::MPS, dir::String)
    nj_exp_path = joinpath(dir, "nj_exp.csv")
    open(nj_exp_path, "w") do io
        header = "j,val"
        println(io, header)
    end

    nj_exp = expect(psi, "N")
    njsum = sum(nj_exp)
    open(nj_exp_path, "a") do io
        for (j, val) in enumerate(nj_exp)
            @printf(io, "%d,%.10f\n", j, val)
        end
        @printf(io, "%d,%.10f\n", -1, njsum)
    end
end

function calc_nupj_expectation(psi::MPS, dir::String)
    nupj_exp_path = joinpath(dir, "nupj_exp.csv")
    open(nupj_exp_path, "w") do io
        header = "j,val"
        println(io, header)
    end

    nupj_exp = expect(psi, "Nup")
    nupjsum = sum(nupj_exp)
    open(nupj_exp_path, "a") do io
        for (j, val) in enumerate(nupj_exp)
            @printf(io, "%d,%.10f\n", j, val)
        end
        @printf(io, "%d,%.10f\n", -1, nupjsum)
    end
end

function calc_ndnj_expectation(psi::MPS, dir::String)
    ndnj_exp_path = joinpath(dir, "ndnj_exp.csv")
    open(ndnj_exp_path, "w") do io
        header = "j,val"
        println(io, header)
    end

    ndnj_exp = expect(psi, "Ndn")
    ndnjsum = sum(ndnj_exp)
    open(ndnj_exp_path, "a") do io
        for (j, val) in enumerate(ndnj_exp)
            @printf(io, "%d,%.10f\n", j, val)
        end
        @printf(io, "%d,%.10f\n", -1, ndnjsum)
    end
end

function calc_sz_expectation(psi::MPS, dir::String)
    sj_exp_path = joinpath(dir, "sx_exp_path.csv")
    open(sj_exp_path, "w") do io
        header = "j,sz"
        println(io, header)
    end

    sz_exp = expect(psi, "Sz")
    szsum = sum(sz_exp)
    open(sj_exp_path, "a") do io
        for j in eachindex(sx_exp)
            @printf(
                io, "%d,%.10f\n",
                j, sz_exp[j],
            )
        end

        @printf(
            io, "%d,%.10f\n",
            -1, szsum,
        )
    end
end

function calc_entanglement_spectrum!(psi::MPS, dir::String)
    es_path = joinpath(dir, "es.csv")

    N = length(psi)
    center_bond = N ÷ 2
    orthogonalize!(psi, center_bond)

    left_inds = (center_bond == 1) ? (siteind(psi, center_bond),) :
                (linkind(psi, center_bond - 1), siteind(psi, center_bond))
    U, S, V = svd(psi[center_bond], left_inds)

    sv_ind = commonind(U, S)
    sv_ind_right = commonind(S, V)

    open(es_path, "w") do io
        write(io, "N,G,val\n")
        offset = 0
        for qn_space in space(sv_ind)
            qn_val = qn_space.first
            qn_n, qn_g = val(qn_val, "N"), val(qn_val, "G")
            block_dim = qn_space.second
            #println("QN = $qn_val, dim = $block_dim")
            for k in 1:block_dim
                sv = S[sv_ind=>offset+k, sv_ind_right=>offset+k]
                prob = sv^2
                @printf(io, "%d,%d,%.10f\n", qn_n, qn_g, -log(prob))
            end
            offset += block_dim
        end
    end

end
function calc_sisj_correlations(psi::MPS, dir::String)
    corr_path = joinpath(dir, "sisj_corr.csv")

    # 2点相関
    spsm = correlation_matrix(psi, "Splus", "Sminus")
    smsp = correlation_matrix(psi, "Sminus", "Splus")
    szsz = correlation_matrix(psi, "Sz", "Sz")

    open(corr_path, "w") do io
        println(io, "i,j,spsm,smsp,szsz")

        for i in 1:length(psi)-1
            for j in i+1:length(psi)
                @printf(
                    io, "%d,%d,%.12e,%.12e,%.12e\n",
                    i, j,
                    real(spsm[i, j]), real(smsp[i, j]), real(szsz[i, j]),
                 )
            end
        end
    end

end

# ---DMRG 実行-----------------------------------------------------------------------
function run_dmrg(N::Int; aspect::Float64, maxdim::Int, eps::Float64, system::String, conserved_nf::Bool=true, conserved_sz::Bool=true, conserved_g::Bool=true, res_dir::String, init_state::Int=1)
    @info "Building sites..."
    sites = build_fqh_nsg_sites(N; conserved_nf=conserved_nf, conserved_sz=conserved_sz, conserved_g=conserved_g, system=system)

    @info "Building initial MPS..."
    psi0 = build_initial_mps(N, sites, init_state)
    @info "Initial MPS QN: $(totalqn(psi0))"

    @info "Building Hamiltonian MPO..."
    H = build_hamiltonian(N, aspect, sites, eps; system=system)

    sweeps = make_sweeps(maxdim)
    @show sweeps

    @info "Starting DMRG..."
    tmp_output_dir = abspath("tmpout")
    if ispath(tmp_output_dir) # refresh
        rm(tmp_output_dir; recursive=true, force=true)
    end
    mkpath(tmp_output_dir)
    observer = MemoryDMRGObserver(eps; dir=res_dir, write_path=tmp_output_dir)
    energy, psi = dmrg(H, psi0, sweeps;
        observer=observer,
        outputlevel=1,
        write_when_maxdim_exceeds=4000,
        write_path=tmp_output_dir
    )

    @info "Calculating expectation values..."
    calc_orbital_occupation_expectations(psi, res_dir)
    calc_orbital_occupation_expectations(psi0, res_dir, file="orb_occ.psi0.csv")
    # calc_nj_expectation(psi, res_dir)
    # calc_nupj_expectation(psi, res_dir)
    # calc_ndnj_expectation(psi, res_dir)
    calc_entanglement_spectrum!(psi, res_dir)
    # calc_sj_expectation(psi, res_dir)
    
    calc_sisj_correlations(psi, res_dir)

    return energy, psi
end
# ---entry point-----------------------------------------------------------------------
let
    N = 120
    aspect = 3.0 * N / 120 
    aspect = 2.0
    maxdim = 1000
    eps = 1e-5
    # conserved_nf = true
    # conserved_sz = true
    # conserved_g = true
    conserved_nf = true
    conserved_sz = true
    conserved_g = true
    system = "edge" # "torus" or "edge"
    # init_state = 1 # 1: |010010...⟩, 2: |020020...⟩, 3: |03000003...⟩
    init_state = 1

    results_dir = abspath("results/fqh_ng_cylinder")
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
    res_dir = results_dir * "/" * timestamp
    mkpath(res_dir)

    config_path = joinpath(res_dir, "config.txt")
    open(config_path, "w") do io
        println(io, "N = $N")
        println(io, "aspect = $aspect")
        println(io, "maxdim = $maxdim")
        println(io, "eps = $eps")
        println(io, "conserved_nf = $conserved_nf")
        println(io, "conserved_g = $conserved_g")
        println(io, "system = $system")
        println(io, "init_state = $init_state")
    end

    energy, psi = run_dmrg(N; aspect=aspect, maxdim=maxdim, eps=eps, system=system, conserved_nf=conserved_nf, conserved_sz=conserved_sz, conserved_g=conserved_g, res_dir=res_dir, init_state=init_state)
    @info @sprintf("Final energy: %.10f", energy)
end
