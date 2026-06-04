using ITensorMPS, ITensorMPOConstruction

# -- coefficients used by build_hamiltonian 
function calc_fqh_coeff_edge(k_diff::Int, l_diff::Int, N::Int, aspect::Float64, threshold::Float64=1e-15)::Float64

    function calc_coeff_tmp(qq_n::Float64)::Float64
        return exp(-0.5 * qq_n) / sqrt(qq_n)
    end

    fqh_coeff = 0.0

    qy1 = (2.0 * π * aspect * l_diff^2) / N
    qy2 = (2.0 * π * aspect * (k_diff + l_diff)^2) / N

    for i in 0:typemax(Int)
        q_x = (2.0 * π * i^2) / (aspect * N)
        qq_1 = q_x + qy1
        qq_2 = q_x + qy2

        if i == 0
            (qq_1 != 0.0) && (fqh_coeff += calc_coeff_tmp(qq_1))
            fqh_coeff -= calc_coeff_tmp(qq_2)
        else
            c1 = calc_coeff_tmp(qq_1)
            c2 = calc_coeff_tmp(qq_2)
            fqh_coeff += 2.0 * c1 * cos((2.0 * π * i * (k_diff + l_diff)) / N)
            fqh_coeff -= 2.0 * c2 * cos((2.0 * π * i * l_diff) / N)
            (c1 < threshold && c2 < threshold) && break
        end
    end


    return fqh_coeff / N
end
function calc_fqh_coeff_torus(k_diff::Int, l_diff::Int, N::Int, aspect::Float64, threshold::Float64=1e-15, iy_unit_cutoff::Int=30)::Float64

    function calc_coeff_tmp(qq_n::Float64)::Float64
        return exp(-0.5 * qq_n) / sqrt(qq_n)
    end

    fqh_coeff = 0.0
    for iy_unit in -iy_unit_cutoff:iy_unit_cutoff
        qy1 = (2.0 * π * aspect * (l_diff + iy_unit * N)^2) / N
        qy2 = (2.0 * π * aspect * (k_diff + l_diff + iy_unit * N)^2) / N

        for i in 0:typemax(Int)
            q_x = (2.0 * π * i^2) / (aspect * N)
            qq_1 = q_x + qy1
            qq_2 = q_x + qy2

            if i == 0
                (qq_1 != 0.0) && (fqh_coeff += calc_coeff_tmp(qq_1))
                fqh_coeff -= calc_coeff_tmp(qq_2)
            else
                c1 = calc_coeff_tmp(qq_1)
                c2 = calc_coeff_tmp(qq_2)
                fqh_coeff += 2.0 * c1 * cos((2.0 * π * i * (k_diff + l_diff)) / N)
                fqh_coeff -= 2.0 * c2 * cos((2.0 * π * i * l_diff) / N)
                (c1 < threshold && c2 < threshold) && break
            end
        end
    end

    return fqh_coeff / N
end
function calc_fqh_coeff(k_diff::Int, l_diff::Int, N::Int, aspect::Float64; system::String="torus")::Float64
    if system == "torus"
        return calc_fqh_coeff_torus(k_diff, l_diff, N, aspect)
    elseif system == "edge"
        return calc_fqh_coeff_edge(k_diff, l_diff, N, aspect)
    else
        error("Invalid system type: $system. Use 'torus' or 'edge'.")
    end
end

# ---------- ハミルトニアン構築 ----------------------------------------------
function build_hamiltonian(N::Int, aspect::Float64, sites, eps; system::String="torus")::MPO
    os = OpSum()
    tol = eps * 1e-4 / N

    # 密度-密度項
    # === Up-Up ===
    for j1 in 1:N-1
        for j2 in j1+1:N
            coeff = calc_fqh_coeff(j2 - j1, 0, N, aspect, system=system)
            if abs(coeff) < tol
                continue
            end
            os .+= coeff, "Nup", j1, "Nup", j2
        end
    end
    # === Down-Down ===
    for j1 in 1:N-1
        for j2 in j1+1:N
            coeff = calc_fqh_coeff(j2 - j1, 0, N, aspect, system=system)
            if abs(coeff) < tol
                continue
            end
            os .+= coeff, "Ndn", j1, "Ndn", j2
        end
    end
    # === Up-Down ===[TODO]: Fix!
    for j1 in 1:N-1
        for j2 in j1+1:N # j1にしてもいい
            coeff = calc_fqh_coeff(j2 - j1, 0, N, aspect, system=system)
            if abs(coeff) < tol
                continue
            end
            os .+= coeff, "Nup", j1, "Ndn", j2
            os .+= coeff, "Ndn", j1, "Nup", j2
        end
    end
    # CAAC 4体項: 
    # === Up-Up ===
    for j1 in 1:N-3
        for j2 in j1+1:N-2
            l_diff = j2 - j1
            for j3 in j2+1:N-1
                j4 = j3 + l_diff
                j4 > N && continue
                k_diff = j3 - j2
                coeff = calc_fqh_coeff(k_diff, l_diff, N, aspect, system=system)
                if abs(coeff) < tol
                    continue
                end
                os .+= -coeff, "Cdagup", j1, "Cup", j2, "Cup", j3, "Cdagup", j4
                os .+= -coeff, "Cup", j1, "Cdagup", j2, "Cdagup", j3, "Cup", j4
            end
        end
    end
    # === Down-Down ===
    for j1 in 1:N-3
        for j2 in j1+1:N-2
            l_diff = j2 - j1
            for j3 in j2+1:N-1
                j4 = j3 + l_diff
                j4 > N && continue
                k_diff = j3 - j2
                coeff = calc_fqh_coeff(k_diff, l_diff, N, aspect, system=system)
                if abs(coeff) < tol
                    continue
                end
                os .+= -coeff, "Cdagdn", j1, "Cdn", j2, "Cdn", j3, "Cdagdn", j4
                os .+= -coeff, "Cdn", j1, "Cdagdn", j2, "Cdagdn", j3, "Cdn", j4
            end
        end
    end
    # === Up-Down === [TODO]: Fix
    for j1 in 1:N-3
        for j2 in j1+1:N-2
            l_diff = j2 - j1
            for j3 in j2+1:N-1
                j4 = j3 + l_diff
                j4 > N && continue
                k_diff = j3 - j2
                coeff = calc_fqh_coeff(k_diff, l_diff, N, aspect, system=system)
                if abs(coeff) < tol
                    continue
                end
                os .+= -coeff, "Cdagup", j1, "Cup", j2, "Cdn", j3, "Cdagdn", j4
                os .+= -coeff, "Cdn", j1, "Cdagdn", j2, "Cdagup", j3, "Cup", j4
            end
        end
    end

    if system == "torus"
        # CCAA 4体項:
        # === Up-Up ===
        for j1 in 1:N-3
            for j2 in j1+1:N-2
                for j3 in j2+1:N-1
                    l_diff = j3 - j2
                    j4 = j1 + N - l_diff
                    j4 > N && continue
                    j4 <= j3 && continue
                    k_diff = j4 - j3
                    coeff = calc_fqh_coeff(k_diff, l_diff, N, aspect, system=system)
                    if abs(coeff) < tol
                        continue
                    end
                    os .+= coeff, "Cdagup", j1, "Cdagup", j2, "Cup", j3, "Cup", j4
                    os .+= coeff, "Cup", j1, "Cup", j2, "Cdagup", j3, "Cdagup", j4
                end
            end
        end
        # === Down-Down ===
        for j1 in 1:N-3
            for j2 in j1+1:N-2
                for j3 in j2+1:N-1
                    l_diff = j3 - j2
                    j4 = j1 + N - l_diff
                    j4 > N && continue
                    j4 <= j3 && continue
                    k_diff = j4 - j3
                    coeff = calc_fqh_coeff(k_diff, l_diff, N, aspect, system=system)
                    if abs(coeff) < tol
                        continue
                    end
                    os .+= coeff, "Cdagdn", j1, "Cdagdn", j2, "Cdn", j3, "Cdn", j4
                    os .+= coeff, "Cdn", j1, "Cdn", j2, "Cdagdn", j3, "Cdagdn", j4
                end
            end
        end
        # === Up-Down === [TODO]: Fix
        # for j1 in 1:N-3
        #     for j2 in j1+1:N-2
        #         for j3 in j2+1:N-1
        #             l_diff = j3 - j2
        #             j4 = j1 + N - l_diff
        #             j4 > N && continue
        #             j4 <= j3 && continue
        #             k_diff = j4 - j3
        #             coeff = calc_fqh_coeff(k_diff, l_diff, N, aspect, system=system)
        #             if abs(coeff) < tol
        #                 continue
        #             end
        #             os .+= coeff, "Cdagdn", j1, "Cdagup", j2, "Cup", j3, "Cdn", j4
        #             os .+= coeff, "Cdn", j1, "Cup", j2, "Cdagup", j3, "Cdagdn", j4
        #         end
        #     end
        # end
    end

    basis_ops = ["I", "Nup", "Ndn", "Cdagup", "Cup", "Cdagdn", "Cdn", "F"]
    op_cache_vec = to_OpCacheVec(sites, [basis_ops for _ in 1:N])

    H = MPO_new(os, sites; basis_op_cache_vec=op_cache_vec, check_for_errors=false)
    max_link_dim = maximum([dim(linkind(H, b)) for b in 1:length(H)-1])
    @info "Constructed Hamiltonian MPO, max link dim = $max_link_dim"

    ITensorMPS.truncate!(H; cutoff=1e-12)

    @info "truncated MPO INFO:"
    @info "  length = $(length(H))"
    link_dims = [dim(linkind(H, b)) for b in 1:length(H)-1]
    @info "  link dims = $link_dims"
    @info "  max link dim = $(maximum(link_dims))"

    return H
end
