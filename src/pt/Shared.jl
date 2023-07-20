"""
Information shared by all processes involved in 
a round of distributed parallel tempering. 
This is updated between rounds but only read during 
a round. 

Fields:
$FIELDS

Only one instance maintained per process. 
"""
@auto struct Shared
    """
    See [`Iterators`](@ref).
    """
    iterators

    """
    See [`tempering`](@ref).
    """
    tempering

    """
    See [`explorer`](@ref).
    """
    explorer

    """
    See [`variational`](@ref).
    """
    variational

    """
    See [`Indexer`](@ref).
    """
    indexer
end

"""
$SIGNATURES 
Create a [`Shared`](@ref) struct based on an [`Inputs`](@ref). 
Uses [`create_tempering()`](@ref) and [`create_explorer()`](@ref).
"""
function Shared(inputs)
    iterators = Iterators() 
    tempering = create_tempering(inputs)
    explorer = create_explorer(inputs) 
    variational = create_variational(inputs)
    indexer = create_replica_indexer(tempering)
    return Shared(iterators, tempering, explorer, variational, indexer)
end

