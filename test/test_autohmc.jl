pt = pigeons(
    target = toy_mvn_target(1), 
    n_chains = 1, 
    explorer = AutoHMC(n_leapfrog = 1, base_n_refresh = 1, exponent_n_refresh = 0.0), 
    # explorer = AutoMALA(base_n_refresh = 1, exponent_n_refresh = 0.0), 
    record = [traces, online], 
    n_rounds = 12
)

println(sample_names(pt))
println(mean(pt)) 
println(var(pt))