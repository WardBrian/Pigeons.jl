"""
Run from runtests.jl
"""

function test_var_reference_Turing()
    model = flip_model_unidentifiable()
    
    # Check NoVarReference()
    inputs = Inputs(
        target = TuringLogPotential(model),
        n_chains = 10,
        n_chains_var_reference = 0,
        seed = 1
    )
    @test_nowarn pt = pigeons(inputs)
    
    # Check GaussianReference()
    inputs = Inputs(
        target = TuringLogPotential(model),
        n_chains = 0,
        n_chains_var_reference = 10,
        var_reference = GaussianReference(),
        seed = 1
    )
    pt = pigeons(inputs)
    # check that a variational reference is indeed used
    @assert pt.shared.tempering.path.ref isa GaussianReference
end


function test_var_reference()
    test_var_reference_Turing()
end