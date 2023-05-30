function compare_pts(p1, p2) 
    @test p1.replicas == p2.replicas 
    @test p1.shared == p2.shared 
    @test p1.reduced_recorders == p2.reduced_recorders 
end

@testset "Checkpoints" begin
    
    p1 = pigeons(target = toy_mvn_target(2), checkpoint = true) 
    p2 = PT("results/latest")
    compare_pts(p1, p2)

    r = pigeons(target = toy_mvn_target(2), checkpoint = true, on = ChildProcess(n_local_mpi_processes = 2))
    p3 = load(r) 
    compare_pts(p1, p3)
end