using Pigeons
using Test
using Distributions
using Random
using OnlineStats
using SplittableRandoms
import Pigeons: mpi_test, my_global_indices, LoadBalance, my_load,
                find_process, split_slice

include("slice_sampler_test.jl")


function test_load_balance(n_processes, n_tasks)
    for p in 1:n_processes
        lb = LoadBalance(p, n_processes, n_tasks)        
        globals = my_global_indices(lb)
        @assert length(globals) == my_load(lb)
        for g in globals
            @assert find_process(lb, g) == p
        end
    end
end

@testset "Stepping stone" begin
    pt = pigeons(target = toy_mvn_target(100));
    p = stepping_stone_pair(pt)
    truth = Pigeons.analytic_lognormalization(toy_mvn_target(100))
    @test abs(p[1] - truth) < 1
    @test abs(p[2] - truth) < 1
end


@testset "Parallelism Invariance" begin
    n_mpis = Sys.iswindows() ? 1 : 4 # MPI on child process crashes on windows;  see c016f59c84645346692f720854b7531743c728bf
    # Turing:
    pigeons(
        target = TuringLogPotential(Pigeons.flip_model_unidentifiable()), 
        n_rounds = 4,
        checked_round = 3, 
        multithreaded = true, 
        checkpoint = true, 
        on = ChildProcess(
                n_local_mpi_processes = n_mpis,
                n_threads = 2))
    # Blang:
    if !Sys.iswindows() # JNI crashes on windows; see commit right after c016f59c84645346692f720854b7531743c728bf
        Pigeons.setup_blang("blangDemos")
        pigeons(
            target = Pigeons.blang_ising(), 
            n_rounds = 4,
            checked_round = 3, 
            multithreaded = true, 
            checkpoint = true, 
            on = ChildProcess(
                    n_local_mpi_processes = n_mpis,
                    n_threads = 2))
    end
    # NB: toy MVN already tested in the doc 
end

@testset "Entanglement" begin
    mpi_test(1, "entanglement_test.jl")
    mpi_test(2, "entanglement_test.jl")

    mpi_test(1, "reduce_test.jl")
    mpi_test(2, "reduce_test.jl")
    mpi_test(3, "reduce_test.jl")
end

@testset "PermutedDistributedArray" begin
    mpi_test(1, "permuted_test.jl", options = ["-s"])
    mpi_test(1, "permuted_test.jl")
    mpi_test(2, "permuted_test.jl")
end

@testset "LoadBalance" begin
    for i in 1:20
        for j in i:30
            test_load_balance(i, j)
        end
    end
end

@testset "LogSum" begin
    m = Pigeons.LogSum()
    
    fit!(m, 2.1)
    fit!(m, 4)
    v1 = value(m)
    @assert v1 ≈ log(exp(2.1) + exp(4))


    fit!(m, 2.1)
    fit!(m, 4)
    m2 = Pigeons.LogSum() 
    fit!(m2, 50.1)
    combined = merge(m, m2)
    @assert value(combined) ≈ log(exp(v1) + exp(50.1))

    fit!(m, 2.1)
    fit!(m, 4)
    empty!(m)
    @assert value(m) == -Pigeons.inf(0.0)
end

function test_split_slice()
    # test disjoint random streams
    set = Set{Float64}()
    push!(set, test_split_slice_helper(1:10)...)
    push!(set, test_split_slice_helper(11:20)...)
    @test length(set) == 20

    # test overlapping
    set = Set{Float64}()
    push!(set, test_split_slice_helper(1:15)...)
    push!(set, test_split_slice_helper(10:20)...)
    @test length(set) == 20
    return true
end

test_split_slice_helper(range) = [rand(r) for r in split_slice(range,  SplittableRandom(1))]

@testset "split_test" begin
    test_split_slice()
end

@testset "Serialize" begin
    mpi_test(1, "serialization_test.jl")
end

@testset "SliceSampler" begin
    test_slice_sampler()
end