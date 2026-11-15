# Run from the 2d-fqh-spin project root:
# julia --project test_fermion4_gr.jl
using ITensors
using ITensorMPS
include("sites_mps.jl")
include("calc_fermion4_gr.jl")
test_fermion4_gr()
