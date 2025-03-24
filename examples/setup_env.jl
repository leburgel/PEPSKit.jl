using Pkg: Pkg
Pkg.activate(@__DIR__)

# develop PEPSKit.jl
Pkg.develop(; path=joinpath(@__DIR__, ".."))

# custom dependencies
Pkg.add(; url="https://github.com/leburgel/OptimKit.jl", rev="lb/hack_backtracking")

# specific versions

# other dependencies
Pkg.add("ArgParse")
Pkg.add("ChainRulesCore")
Pkg.add("JLD2")
Pkg.add("Revise")
Pkg.add("LinearAlgebra")
Pkg.add("TensorKit")
Pkg.add("MPSKit")
Pkg.add("MPSKitModels")
Pkg.add("OptimKit")
Pkg.add("KrylovKit")
Pkg.add("QuadGK")
Pkg.add("Revise")
