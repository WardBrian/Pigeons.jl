using Pigeons
using BridgeStan
const BS = BridgeStan
using Random 
using SplittableRandoms
using Plots

# function main()
    # settings
    bernoulli_stan = "test/nikola_temp/bernoulli.stan"
    bernoulli_data = "test/nikola_temp/bernoulli.data.json"

    # PT settings
    n_rounds = 10
    n_chains = 10

    # create Stan models
    slp = StanLogPotential(bernoulli_stan, bernoulli_data)

    # run Pigeons
    pt = pigeons(
        target = slp, 
        n_rounds = n_rounds, 
        n_chains = n_chains, 
        recorder_builders = [traces], 
        n_chains_var_reference = n_chains, 
        explorer = AutoMALA(),
        var_reference = GaussianReference()
    )
    s = get_sample(pt, n_chains)
    samples_vec = map((state) -> state.x[1], s)
    p = Plots.histogram(samples_vec, bins = 0:0.05:1)
    # display(p)
    # nothing
# end

# main()