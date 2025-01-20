using Pkg: Pkg
Pkg.activate(@__DIR__)

Pkg.develop(; path=joinpath(@__DIR__, ".."))
Pkg.add("Revise")
Pkg.add("LinearAlgebra")
Pkg.add("TensorKit")
Pkg.add("MPSKit")
Pkg.add("MPSKitModels")
Pkg.add("OptimKit")
Pkg.add("KrylovKit")
Pkg.add("QuadGK")
