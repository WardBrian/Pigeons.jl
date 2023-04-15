@concrete struct FusedSwap
    log_potentials 
    cdfs::Vector 
    icdfs::Vector
end

FusedSwap(log_potentials) = FusedSwap(log_potentials, [], [])

function adapt_pair_swapper(::FusedSwap, pt, updated_tempering)
    log_potentials = updated_tempering.log_potentials 
    cdfs = []
    icdfs = [] 
    grids = updated_tempering.schedule.grids
    for i in eachindex(grids)
        beta = grids[i] 
        points, cumulative_prs = 
            interpolated_log_potential_distribution(pt, beta)
        push!(cdfs, interpolate_cdf(points, cumulative_prs))
        push!(icdfs, interpolate_cdf(points, cumulative_prs, true))
    end
    return FusedSwap(log_potentials, cdfs, icdfs)
end

struct FusedStat 
    log_ratio::Float64 
    uniform::Float64
    proposed::Float64
end

const fused_swap_tol = Ref(1e-5)

function height_mover(pair_swapper, my_chain, partner_chain)
    if isempty(pair_swapper.cdfs)
        id_fct(x) = x 
        one_fct(x) = 1
        return id_fct, one_fct
    end
    T = pair_swapper.icdfs[partner_chain] ∘ pair_swapper.cdfs[my_chain]
    dT(x) = ForwardDiff.derivative(T, x)
    return T, dT
end

function swap_stat(pair_swapper::FusedSwap, replica::Replica, partner_chain::Int) 
    # everythig will be in place, so save current location along the orbit
    current_t, mover = state_mover(pair_swapper, replica)

    T, dT = height_mover(pair_swapper, replica.chain, partner_chain)
    W_mine,  dW_mine  = log_density_slice(mover, pair_swapper.log_potentials[replica.chain])
    W_yours, dW_yours = log_density_slice(mover, pair_swapper.log_potentials[partner_chain]) 

    current_height  = W_mine(current_t)
    proposed_height = T(current_height)


    # compute "pre-involution" i.e. proposed state move s.t. we have not checked yet that the involutive property holds
    if isnan(proposed_height)
        proposed_t = NaN 
    else
        proposed_t     = pre_involution(W_yours, dW_yours, current_t,  proposed_height) 
    end

    if isnan(proposed_t) # i.e. root finding in pre_involution failed
        fused = false
    else
        reversed_t = pre_involution(W_mine,  dW_mine,  proposed_t, current_height)
        checked = isapprox(current_t, reversed_t; atol = fused_swap_tol[]) # if reversed_t is NaN, 'checked' and hence 'fused' will be false
        fused = checked && proposed_t != current_t # are use doing a 'fused move' (where both x and beta change)? otherwise, classical swap where only beta's are exchanged
    end

    # go back to current point now that done with up to 2 pre_involution() calls, 
    # will do the actual moving after the accept-reject step
    move!(mover, current_t)
    @assert W_mine(current_t) ≈ current_height

    if fused
        log_ratio = logabs(dW_mine(current_t)) - logabs(dW_yours(proposed_t)) + logabs(dT(current_height))
        return FusedStat(log_ratio, rand(replica.rng), proposed_t)
    else # fuse line search failed... then use classical ratio:
        log_ratio = log_unnormalized_ratio(pair_swapper.log_potentials, partner_chain, replica.chain, replica.state)
        @assert !isnan(log_ratio) "$(replica.state)"
        return FusedStat(log_ratio, rand(replica.rng), current_t)
    end
end

logabs(x) = log(abs(x))

function pre_involution(W, dW, start_point, proposed_height)
    shifted_W(x) = W(x) - proposed_height
    problem = ZeroProblem((shifted_W, dW), start_point)
    return solve(problem, atol = fused_swap_tol[] / 2.0)
end

@concrete mutable struct StateMover
    current_t
    replica
end

function state_mover(pair_swapper::FusedSwap, replica)
    # consider set of point {exp(t) x : t in Real}
    current_t = 0.0
    mover = StateMover(current_t, replica)
    return current_t, mover
end

function move!(mover::StateMover, to)
    if abs(to) > 10 
        return 
    end
    if to == mover.current_t
        return 
    end
    # current replica state is cur = exp(t) x
    # you want to got to       new = exp(t') x
    # we have: new = exp(t') x = (exp(t) / exp(t)) exp(t') x = (exp(t') / exp(t)) ( exp(t) x ) = (exp(t' - t)) cur
    multiplier = exp(to - mover.current_t)
    mover.replica.state *= multiplier
    mover.current_t = to
end


function log_density_slice(mover, log_potential)
    function W(t)
        move!(mover, t) 
        return log_potential(mover.replica.state)
    end 
    function dW(t)
        move!(mover, t)
        # since the derivative of exp(t) is exp(t) this is just the directional derivative at x_t and along x_t
        return directional_derivative(log_potential, mover.replica.state, mover.replica.state) 
    end
    return W, dW
end

function directional_derivative(log_potential, x, v)
    # NB: may want to use FwdDiff with a given direction
    # seems implemented in https://github.com/JuliaDiff/SparseDiffTools.jl/blob/master/src/differentiation/jaches_products.jl#L3-L13
    # but does not have public API at the moment??? https://github.com/JuliaDiff/ForwardDiff.jl/issues/319
    grad = gradient(log_potential, x)
    return dot(grad, v)
end

function record_swap_stats!(pair_swapper::FusedSwap, recorders, chain1::Int, stat1, chain2::Int, stat2)
    acceptance_pr = swap_acceptance_probability(stat1, stat2)
    key1 = (chain1, chain2)
    
    record_if_requested!(recorders, :swap_acceptance_pr, (key1, acceptance_pr))
    
    # TODO: derive new normalization constant identity
    # key2 = (chain2, chain1)
    # record_if_requested!(recorders, :log_sum_ratio, (key1, stat1.log_ratio))
    # record_if_requested!(recorders, :log_sum_ratio, (key2, stat2.log_ratio)) # compute both to estimate a sandwich
end

function apply_swap!(pair_swapper::FusedSwap, partner_chain::Int, do_swap::Bool, replica, my_swap_stat)
    if do_swap
        replica.chain = partner_chain 
        _, mover = state_mover(pair_swapper, replica)
        move!(mover, my_swap_stat.proposed)
    end
end