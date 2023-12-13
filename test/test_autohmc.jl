using Plots 

pt = pigeons(
    target = toy_mvn_target(2), 
    # target = Pigeons.stan_funnel(9), 
    n_chains = 1, 
    explorer = AutoHMC(n_leapfrog = 5, base_n_refresh = 1, exponent_n_refresh = 0.0), 
    # explorer = AutoMALA(base_n_refresh = 1, exponent_n_refresh = 0.0), 
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

using Base.Threads
# run pi-invariance test for Alex
function main() 
    # simulation settings 
    n_sim = 10_000 

    # run simulation 
    margin1 = Vector{Float64}(undef, n_sim)
    Threads.@threads for i in 1:n_sim
        pt = pigeons(
            target = toy_mvn_target(2), 
            n_chains = 1, 
            explorer = AutoHMC(n_leapfrog = 20, base_n_refresh = 1, exponent_n_refresh = 0.0), 
            record = [traces, online], 
            n_rounds = 2, 
            show_report = false, 
            seed = i
        )
        samples = get_sample(pt) 
        margin1[i] = samples[end][1]
    end
    println(mean(margin1))
    println(var(margin1))
    # println(margin1)
end 

main()