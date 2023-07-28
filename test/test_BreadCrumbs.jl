using MCMCChains

@testset "Multivariate BreadCrumbs" begin
    function unid_log_potential(x; n_trials=100, n_successes=50) 
        p = prod(x)
        return n_successes*log(p) + (n_trials-n_successes)*log1p(-p)
    end
    ref_dist = product_distribution(Uniform(), Uniform())
    pt = pigeons(
        BreadCrumbs(unid_log_potential, ref_dist),
        n_rounds = 12,
        record = [traces]
    )

    # collect the statistics and convert to MCMCChains' Chains
    samples = Chains(sample_array(pt), variable_names(pt))
end