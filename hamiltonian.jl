using ITensorMPS, ITensorMPOConstruction

# -- coefficients used by build_hamiltonian 
function calc_form_factor(qq_n::Float64)::Float64
    return exp(-0.5 * qq_n) / sqrt(qq_n)
end

function calc_akl_edge(k_diff::Int, l_diff::Int, N::Int, aspect::Float64, threshold::Float64=1e-15)::Float64
    insulator_layer = 0
    insulator_layer += 1
    N *= insulator_layer
    aspect *= insulator_layer

    akl = 0.0

    q_y = (2.0 * π * aspect * l_diff^2) / N

    for i in 0:typemax(Int)
        q_x = (2.0 * π * i^2) / (aspect * N)
        qq = q_x + q_y

        if i == 0
            (qq != 0.0) && (akl += calc_form_factor(qq))
        else
            ff = calc_form_factor(qq)
            akl += 2.0 * ff * cos((2.0 * π * i * (k_diff + l_diff)) / N)
            ff < threshold && break
        end
    end

    return akl / N
end

function calc_akl_torus(k_diff::Int, l_diff::Int, N::Int, aspect::Float64, threshold::Float64=1e-15, iy_unit_cutoff::Int=30)::Float64

    akl = 0.0
    for iy_unit in -iy_unit_cutoff:iy_unit_cutoff
        q_y = (2.0 * π * aspect * (l_diff + iy_unit * N)^2) / N

        for i in 0:typemax(Int)
            q_x = (2.0 * π * i^2) / (aspect * N)
            qq = q_x + q_y

            if i == 0
                (qq != 0.0) && (akl += calc_form_factor(qq))
            else
                ff = calc_form_factor(qq)
                akl += 2.0 * ff * cos((2.0 * π * i * (k_diff + l_diff)) / N)
                ff < threshold && break
            end
        end
    end

    return akl / N
end

function calc_bkl_ferro_edge(k_diff::Int, l_diff::Int, N::Int, aspect::Float64)::Float64
    return calc_akl_edge(k_diff, l_diff, N, aspect) - calc_akl_edge(-k_diff, k_diff + l_diff, N, aspect)
end
function calc_bkl_ferro_torus(k_diff::Int, l_diff::Int, N::Int, aspect::Float64)::Float64
    return calc_akl_torus(k_diff, l_diff, N, aspect) - calc_akl_torus(-k_diff, k_diff + l_diff, N, aspect)
end

function calc_akl_updown(k_diff::Int, l_diff::Int, N::Int, aspect::Float64; system::String="torus")::Float64
    if system == "torus"
        return calc_akl_torus(k_diff, l_diff, N, aspect)
    elseif system == "edge"
        return calc_akl_edge(k_diff, l_diff, N, aspect)
    else
        error("Invalid system type: $system. Use 'torus' or 'edge'.")
    end
end

function calc_bkl_ferro(k_diff::Int, l_diff::Int, N::Int, aspect::Float64; system::String="torus")::Float64
    if system == "torus"
        return calc_bkl_ferro_torus(k_diff, l_diff, N, aspect)
    elseif system == "edge"
        return calc_bkl_ferro_edge(k_diff, l_diff, N, aspect)
    else
        error("Invalid system type: $system. Use 'torus' or 'edge'.")
    end
end

function calc_confining_potential(A::Float64, d::Float64, site::Int, ns::Int)::Float64
    return A * (exp(-(site-1)/d) + exp((site-ns)/d))
end

# ---------- ハミルトニアン構築 ----------------------------------------------
function build_hamiltonian(N::Int, aspect::Float64, sites, eps; system::String="torus")::MPO
    os = OpSum()
    tol = eps * 1e-4 / N

    # B = 0.00

    # # Zeeman term
    # for j1 in 1:N
    #     os .+= -B, Nup, j1
    #     os .+= B, Ndn, j1
    # end

    # 密度-密度項
    # === Up-Up ===
    for j1 in 1:N-1
        for j2 in j1+1:N
            coeff = calc_bkl_ferro(j2 - j1, 0, N, aspect, system=system)
            if abs(coeff) < tol
                continue
            end
            os .+= coeff, "Nup", j1, "Nup", j2
        end
    end
    # === Down-Down ===
    for j1 in 1:N-1
        for j2 in j1+1:N
            coeff = calc_bkl_ferro(j2 - j1, 0, N, aspect, system=system)
            if abs(coeff) < tol
                continue
            end
            os .+= coeff, "Ndn", j1, "Ndn", j2
        end
    end
    # === Up-Down ===[TODO]: Fix!
    for j1 in 1:N-1
        for j2 in j1+1:N
            coeff = calc_akl_updown(j2 - j1, 0, N, aspect, system=system)
            if abs(coeff) < tol
                continue
            end
            os .+= coeff, "Nup", j1, "Ndn", j2
            os .+= coeff, "Ndn", j1, "Nup", j2
        end
    end
    for j1 in 1:N # 同じサイトにupdownがある場合の対角項
        # Akl nup ndn
        coeff = calc_akl_updown(0, 0, N, aspect, system=system)
        if abs(coeff) < tol
            continue
        end
        os .+= coeff, "NupNdn", j1

        # V nup Ndn
        # coeff = calc_confining_potential(3.0, 1.0, j1, N)
        # if abs(coeff) < tol
        #     continue
        # end
        # os .+= coeff, "Nup", j1
        # os .+= coeff, "Ndn", j1
        # os .+= 2 * coeff, "NupNdn", j1
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
                coeff = calc_bkl_ferro(k_diff, l_diff, N, aspect, system=system)
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
                coeff = calc_bkl_ferro(k_diff, l_diff, N, aspect, system=system)
                if abs(coeff) < tol
                    continue
                end
                os .+= -coeff, "Cdagdn", j1, "Cdn", j2, "Cdn", j3, "Cdagdn", j4
                os .+= -coeff, "Cdn", j1, "Cdagdn", j2, "Cdagdn", j3, "Cdn", j4
            end
        end
    end
    # === Up-Down === [TODO]: Fix
    for j1 in 1:N-3 # direct term: u-u-d-d or d-d-u-u
        for j2 in j1+1:N-2
            l_diff = j2 - j1
            for j3 in j2+1:N-1
                j4 = j3 + l_diff
                j4 > N && continue
                k_diff = j3 - j2
                coeff = calc_akl_updown(k_diff, l_diff, N, aspect, system=system)
                if abs(coeff) < tol
                    continue
                end
                os .+= -coeff, "Cdagup", j1, "Cup", j2, "Cdn", j3, "Cdagdn", j4 # u-u-d-d
                os .+= -coeff, "Cup", j1, "Cdagup", j2, "Cdagdn", j3, "Cdn", j4 # h.c.
                os .+= -coeff, "Cdn", j1, "Cdagdn", j2, "Cdagup", j3, "Cup", j4 # d-d-u-u
                os .+= -coeff, "Cdagdn", j1, "Cdn", j2, "Cup", j3, "Cdagup", j4 # h.c.
            end
        end
    end
    for j1 in 1:N-3 # cross term: u-d-u-d or d-u-d-u
        for j2 in j1+1:N-2
            l_diff = j2 - j1
            for j3 in j2+1:N-1
                j4 = j3 + l_diff
                j4 > N && continue
                k_diff = j3 - j2
                coeff = calc_akl_updown(-k_diff, k_diff+l_diff, N, aspect, system=system) # cross -> -k
                if abs(coeff) < tol
                    continue
                end
                os .+= coeff, "Cdagup", j1, "Cdn", j2, "Cup", j3, "Cdagdn", j4 # u-d-u-d
                os .+= coeff, "Cup", j1, "Cdagdn", j2, "Cdagup", j3, "Cdn", j4 # h.c.
                os .+= coeff, "Cdagdn", j1, "Cup", j2, "Cdn", j3, "Cdagup", j4 # d-u-d-u
                os .+= coeff, "Cdn", j1, "Cdagup", j2, "Cdagdn", j3, "Cup", j4 # h.c.
            end
        end
    end
    for j1 in 1:N-1 # cross term: u-d-u-d or d-u-d-u
        for j2 in j1+1:N
            
            k_diff = j2 - j1
            coeff = calc_akl_updown(-k_diff, k_diff, N, aspect, system=system) # cross -> -k
            if abs(coeff) < tol
                continue
            end
            os .+= -coeff, "Splus", j1, "Sminus", j2 # u-d-u-d
            os .+= -coeff, "Sminus", j1, "Splus", j2 # h.c.
        end
    end
    for j1 in 1:N-2 # cross term with same-site-ud: u-ud-d or d-ud-u
        for j2 in j1+1:N-1
            l_diff = j2 - j1
            j4 = j2 + l_diff
            j4 > N && continue
            coeff = calc_akl_updown(0, l_diff, N, aspect, system=system) # cross -> -k
            if abs(coeff) < tol
                continue
            end
            os .+= -coeff, "Cdagup", j1, "CupCdn", j2, "Cdagdn", j4 # u-ud-d
            os .+= -coeff, "Cup", j1, "CdagupCdagdn", j2, "Cdn", j4 # h.c.
            os .+= coeff, "Cdagdn", j1, "CupCdn", j2, "Cdagup", j4 # d-ud-u
            os .+= coeff, "Cdn", j1, "CdagupCdagdn", j2, "Cup", j4 # h.c.
            # os .+= -coeff, "Cdagup", j1, "Cup", j2, "Cdn", j2, "Cdagdn", j4 # u-ud-d
            # os .+= -coeff, "Cup", j1, "Cdagup", j2, "Cdagdn", j2, "Cdn", j4 # h.c.
            # os .+= coeff, "Cdagdn", j1, "Cup", j2, "Cdn", j2, "Cdagup", j4 # d-ud-u
            # os .+= coeff, "Cdn", j1, "Cdagup", j2, "Cdagdn", j2, "Cup", j4 # h.c.
        end
    end

    if system == "torus"
        # CCAA 4体項:
        # === Up-Up ===
        # for j1 in 1:N-3
        #     for j2 in j1+1:N-2
        #         for j3 in j2+1:N-1
        #             l_diff = j3 - j2
        #             j4 = j1 + N - l_diff
        #             j4 > N && continue
        #             j4 <= j3 && continue
        #             k_diff = j4 - j3
        #             coeff = calc_bkl_ferro(k_diff, l_diff, N, aspect, system=system)
        #             if abs(coeff) < tol
        #                 continue
        #             end
        #             os .+= coeff, "Cdagup", j1, "Cdagup", j2, "Cup", j3, "Cup", j4
        #             os .+= coeff, "Cup", j1, "Cup", j2, "Cdagup", j3, "Cdagup", j4
        #         end
        #     end
        # end
        # # === Down-Down ===
        # for j1 in 1:N-3
        #     for j2 in j1+1:N-2
        #         for j3 in j2+1:N-1
        #             l_diff = j3 - j2
        #             j4 = j1 + N - l_diff
        #             j4 > N && continue
        #             j4 <= j3 && continue
        #             k_diff = j4 - j3
        #             coeff = calc_bkl_ferro(k_diff, l_diff, N, aspect, system=system)
        #             if abs(coeff) < tol
        #                 continue
        #             end
        #             os .+= coeff, "Cdagdn", j1, "Cdagdn", j2, "Cdn", j3, "Cdn", j4
        #             os .+= coeff, "Cdn", j1, "Cdn", j2, "Cdagdn", j3, "Cdagdn", j4
        #         end
        #     end
        # end
        # # === Up-Down === [TODO]: Fix
        # for j1 in 1:N-3
        #     for j2 in j1+1:N-2
        #         for j3 in j2+1:N-1
        #             l_diff = j3 - j2
        #             j4 = j1 + N - l_diff
        #             j4 > N && continue
        #             j4 <= j3 && continue
        #             k_diff = j4 - j3
        #             coeff = calc_akl_updown(k_diff, l_diff, N, aspect, system=system)
        #             if abs(coeff) < tol
        #                 continue
        #             end
        #             os .+= coeff, "Cdagdn", j1, "Cdagup", j2, "Cup", j3, "Cdn", j4
        #             os .+= coeff, "Cdn", j1, "Cup", j2, "Cdagup", j3, "Cdagdn", j4
        #         end
        #     end
        # end
    end

    basis_ops = [
        "I", "Nup", "Ndn", "NupNdn", 
        "Cdagup", "Cup", "Cdagdn", "Cdn", 
        "CdagupCdagdn", "CupCdn", 
        "Splus", "Sminus", 
        "F"
    ]
    op_cache_vec = to_OpCacheVec(sites, [basis_ops for _ in 1:N])

    H = MPO_new(os, sites; basis_op_cache_vec=op_cache_vec, check_for_errors=true)
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
