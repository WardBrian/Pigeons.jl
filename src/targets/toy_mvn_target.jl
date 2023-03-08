""" 
$SIGNATURES 

A toy multi-variate normal (mvn) target distribution used for testing. 
Uses a specialized path, [`ScaledPrecisionNormalPath`](@ref), 
such that i.i.d. sampling is possible at all chains (via [`ToyExplorer`](@ref)). 
"""
@provides target toy_mvn_target(dim::Int) = ScaledPrecisionNormalPath(dim) 

create_state_initializer(target::ScaledPrecisionNormalPath, ::Inputs) = target 
initialization(target::ScaledPrecisionNormalPath, rng::SplittableRandom, _::Int64) = 
    zeros(target.dim)

create_explorer(::ScaledPrecisionNormalPath, ::Inputs) = 
    ToyExplorer()

sample_iid!(log_potential::ScaledPrecisionNormalLogPotential, replica) =
    rand!(replica.rng, replica.state, log_potential)

Random.rand!(rng::AbstractRNG, x::AbstractVector, log_potential::ScaledPrecisionNormalLogPotential) =
    for i in eachindex(x)
        x[i] = randn(rng) / sqrt(log_potential.precision)
    end

create_path(target::MultivariateNormal, ::Inputs) = 
    target # a bit of a special case here: the target is also a path