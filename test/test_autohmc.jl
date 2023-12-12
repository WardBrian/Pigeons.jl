using Plots 

pt = pigeons(
    target = toy_mvn_target(2), 
    # target = Pigeons.stan_funnel(9), 
    n_chains = 1, 
    # explorer = AutoHMC(n_leapfrog = 5, base_n_refresh = 1, exponent_n_refresh = 0.0), 
    explorer = AutoMALA(base_n_refresh = 1, exponent_n_refresh = 0.0), 
    record = [traces, online], 
    n_rounds = 10
)

println(sample_names(pt))
println(mean(pt)) 
println(var(pt))
samples = get_sample(pt) 

p = Plots.plot(
    [samples[i][1] for i in eachindex(samples)], 
    [samples[i][2] for i in eachindex(samples)]
)
display(p)