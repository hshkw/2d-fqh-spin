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
            QN(("N", 0)) => 1,   # |empty⟩
            QN(("N", 1)) => 1,   # |up⟩
            QN(("N", 1)) => 1,   # |down⟩
            QN(("N", 2)) => 1,   # |occupied⟩
        end
    else
        return [QN() => 1, QN() => 1]  # No conservation, just two states
    end
end