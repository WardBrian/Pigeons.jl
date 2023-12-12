"""
$SIGNATURES

Hamiltonian Monte Carlo with automatic step size selection and fixed number of leapfrog steps.

AutoHMC is essentially `n_leapfrog` applications of `AutoMALA`, where `n_leapfrog` is the number 
of leapfrog steps. See [`AutoMALA`](@ref) for more information. 
The difference is that *between* leapfrog steps, the momentum is *not* refreshed and
the sign of the momentum is *not* flipped at any point. 
To prove that the algorithm is pi-invariant, one uses skew detailed balance instead of detailed balance. 
The following optional keyword parameters are available:
$FIELDS
"""
@kwdef struct AutoHMC{T,TPrec <: Preconditioner}
    """ 
    Number of leapfrog steps before computing an MH ratio and doing a momentum refreshment.
    """ 
    n_leapfrog::Int = 10 

    """
    The base number of steps (equivalently, momentum refreshments) between swaps.
    This base number gets multiplied by `ceil(Int, dim^(exponent_n_refresh))`.
    This is done after completing the `n_leapfrog` leapfrog steps.
    """
    base_n_refresh::Int = 3

    """
    Used to scale the increase in number of refreshment with dimensionality.
    This is done after completing the `n_leapfrog` leapfrog steps.
    """
    exponent_n_refresh::Float64 = 0.35

    """
    The default backend to use for autodiff.
    See https://github.com/tpapp/LogDensityProblemsAD.jl#backends

    Certain targets may ignore it, e.g. if a manual differential is
    offered or when calling an external program such as Stan.
    """
    default_autodiff_backend::Symbol = :ForwardDiff

    """
    Starting point for the automatic step size algorithm.
    Gets updated automatically between each round.
    """
    step_size::Float64 = 1.0

    """
    A strategy for building a preconditioner.
    """
    preconditioner::TPrec = MixDiagonalPreconditioner()

    """
    This gets updated after first iteration; initially `nothing` in
    which case an identity mass matrix is used.
    """
    estimated_target_std_deviations::T = nothing
end

function adapt_explorer(explorer::AutoHMC, reduced_recorders, current_pt, new_tempering)
    estimated_target_std_deviations = adapt_preconditioner(explorer.preconditioner, reduced_recorders)
    # use the mean across chains of the mean shrink/grow factor to compute a new baseline stepsize
    updated_step_size = explorer.step_size * mean(mean.(values(value(reduced_recorders.am_factors))))
    return AutoHMC(
                explorer.n_leapfrog,
                explorer.base_n_refresh, explorer.exponent_n_refresh, 
                explorer.default_autodiff_backend,
                updated_step_size,
                explorer.preconditioner,
                estimated_target_std_deviations)
end

#=
Extract info common to all types of target and perform a step!()
=#
function _extract_commons_and_run!(explorer::AutoHMC, replica, shared, log_potential, state::AbstractVector)

    log_potential_autodiff = ADgradient(explorer.default_autodiff_backend, log_potential, replica.recorders.buffers)
    is_first_scan_of_round = shared.iterators.scan == 1

    auto_hmc!(
        replica.rng,
        explorer,
        log_potential_autodiff,
        state,
        replica.recorders,
        replica.chain,
        # In the transient phase, the rejection rate for the
        # reversibility check can be high, so skip accept-rejct
        # for the initial scan of each round.
        # We only do this on the first scan of each round.
        # Since the number of iterations per round increases,
        # the fraction of time we do this decreases to zero.
        # !is_first_scan_of_round # debug
        true # debug
    )
end

function auto_hmc!(
        rng::AbstractRNG,
        explorer::AutoHMC,
        target_log_potential,
        state::Vector,
        recorders,
        chain,
        use_mh_accept_reject)

    dim = length(state)

    momentum = get_buffer(recorders.buffers, :am_momentum_buffer, dim)
    start_momentum = get_buffer(recorders.buffers, :am_momentum_buffer, dim) # incremented between leapfrog steps
    first_start_momentum = get_buffer(recorders.buffers, :am_momentum_buffer, dim) # stays at 0th leapfrog step 
    diag_precond = get_buffer(recorders.buffers, :am_ones_buffer, dim)
    build_preconditioner!(diag_precond, explorer.preconditioner, rng, explorer.estimated_target_std_deviations)
    start_state = get_buffer(recorders.buffers, :am_state_buffer, dim) # incremented between leapfrog steps
    first_start_state = get_buffer(recorders.buffers, :am_first_state_buffer, dim) # stays at 0th leapfrog step
    first_start_state .= state

    n_refresh = explorer.base_n_refresh * ceil(Int, dim^explorer.exponent_n_refresh)
    for i in 1:n_refresh # each time do `n_leapfrog` steps
        # println(state) # debug
        randn!(rng, momentum) # refresh momentum
        first_start_momentum .= momentum
        init_joint_log = log_joint(target_log_potential, state, momentum)
        @assert isfinite(init_joint_log) "AutoHMC can only be called on a configuration of positive density."
        final_joint_log = init_joint_log 
        for _ in 1:explorer.n_leapfrog
            start_state .= state # the 'start' is the beginning of this leapfrog step
            start_momentum .= momentum 

            # Randomly pick a "reasonable" range of MH accept probabilities (in log-scale)
            a = rand(rng)
            b = rand(rng)
            lower_bound = log(min(a, b))
            upper_bound = log(max(a, b))

            proposed_exponent =
                auto_step_size(
                    target_log_potential,
                    diag_precond,
                    state, momentum,
                    recorders, chain,
                    explorer.step_size, lower_bound, upper_bound)
            proposed_step_size = explorer.step_size * 2.0^proposed_exponent

            # move to proposed point (single leapfrog application without a momentum flip)
            leap_frog!(
                target_log_potential,
                diag_precond,
                state, momentum, proposed_step_size
            ) 

            momentum .*= -1.0 # flip
            reversed_exponent =
                auto_step_size(
                    target_log_potential,
                    diag_precond,
                    state, momentum,
                    recorders, chain,
                    explorer.step_size, lower_bound, upper_bound)
            momentum .*= -1.0
            if reversed_exponent == proposed_exponent
                final_joint_log = log_joint(target_log_potential, state, momentum)
            else
                # step size reversibility check not satisfied, so go back 
                # to where you were at the beginning of the leapfrog step
                state .= start_state
                momentum .= start_momentum
            end
        end
        if use_mh_accept_reject # applies to the entire trajectory consisting of n_leapfrog steps
            probability =  min(1.0, exp(final_joint_log - init_joint_log))
            @record_if_requested!(recorders, :explorer_acceptance_pr, (chain, probability))
            if rand(rng) < probability
                # accept trajectory: nothing to do, we work in-place
            else
                # reject entire trajectory: go back to the *very first* start state
                state .= first_start_state 
                momentum .= first_start_momentum
            end
        end
    end
end

function explorer_recorder_builders(explorer::AutoHMC)
    result = [
        explorer_acceptance_pr,
        explorer_n_steps,
        am_factors,
        buffers
    ]
    add_precond_recorder_if_needed!(result, explorer)
    return result
end
