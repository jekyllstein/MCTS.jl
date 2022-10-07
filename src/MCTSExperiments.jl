module MCTSExperiments

#repository from book authors https://github.com/algorithmsbooks/DecisionMakingProblems.jl which contains some of the basic functions and data structures and functions in the book as well as game environments
using DecisionMakingProblems
using Base.Threads
using ThreadPools
using Distributed
using Base.Threads
using TailRec
using Transducers

export DecisionMakingProblems, MDP, MonteCarloTreeSearch, MCTSPar, init_MCTSPar, rollout, UCT, LeafP, RootP, TreeP, WU_UCT, VL_UCT_hard, VL_UCT_soft, BU_UCT, clear_dicts!

transition_and_reward = DecisionMakingProblems.transition_and_reward
#goal is to first use the basic functionality in the game2048 forked package and put it into a structure where the code from the algorithms book can be used with it

#the following code is copied from Appendix G.5 of the book found here: https://algorithmsbook.com/
function Base.findmax(f::Function, xs)
    f_max = -Inf
    x_max = first(xs)
    for x in xs
        v = f(x)
        if v > f_max
            f_max, x_max = v, x
        end
    end
    return f_max, x_max
end

Base.argmax(f::Function, xs) = findmax(f, xs)[2]

#the following code is copied from Chapter 7 of the book found here: https://algorithmsbook.com/
# struct MDP
#     γ # discount factor
#     𝒮 # state space
#     𝒜 # action space
#     T # transition function
#     R # reward function
#     TR # sample transition and reward
# end
import DecisionMakingProblems.MDP #same struct as found in their module

function lookahead(𝒫::MDP, U, s, a)
    𝒮, T, R, γ = 𝒫.𝒮, 𝒫.T, 𝒫.R, 𝒫.γ
    return R(s,a) + γ*sum(T(s,a,s′)*U(s′) for s′ in 𝒮)
end

function lookahead(𝒫::MDP, U::Vector, s, a)
    𝒮, T, R, γ = 𝒫.𝒮, 𝒫.T, 𝒫.R, 𝒫.γ
    return R(s,a) + γ*sum(T(s,a,s′)*U[i] for (i,s′) in enumerate(𝒮))
end

struct ValueFunctionPolicy
    𝒫 # problem
    U # utility function
end

function greedy(𝒫::MDP, U, s)
    u, a = findmax(a->lookahead(𝒫, U, s, a), 𝒫.𝒜)
    return (a=a, u=u)
end

#following code is copied from Chapter 9 of the book found here: https://algorithmsbook.com/

#forward search functions
struct RolloutLookahead
    𝒫 # problem
    π # rollout policy
    d # depth
end

randstep(𝒫::MDP, s, a) = 𝒫.TR(s, a)

function rollout(TR::Function, γ::Float64, s, π, d, isterminal::Function = s -> false)
    ret = 0.0
    t = 1
    while !isterminal(s) && (t <= d)
        a = π(s)
        s, r = TR(s, a)
        ret += γ^(t-1) * r
        t += 1
    end
    return (ret, t-1, s)
end

function rollout(TR::Function, γ::Float64, s, π, isterminal::Function = s -> false)
    ret = 0.0
    t = 1
    while !isterminal(s)
        a = π(s)
        s, r = TR(s, a)
        ret += γ^(t-1) * r
        t += 1
    end
    return (ret, t-1, s)
end

function (π::RolloutLookahead)(s)
    U(s) = rollout(π.𝒫, s, π.π, π.d)
    return greedy(π.𝒫, U, s).a
end

#= 
need to understand the convensions in this struct to see how to define all the components
what does 𝒫 need to have?
𝒫.𝒜, 𝒫.TR, 𝒫.γ

what does N need to have?
N[(s,a)], so a dictionary of counts indexed by state/action pairs

what does Q need to have?
Q[(s,a)], so it is the same structure as N, clearly Q is not a function on state/action pairs but just a lookup

d and m should just be integers
=#

struct MonteCarloTreeSearch
    𝒫::MDP # problem
    N::Dict # visit counts
    Q::Dict # action value estimates
    d::Integer # depth
    m::Integer # number of simulations
    c::AbstractFloat # exploration constant
    U::Function # value function estimate
end

abstract type MCTSAlgo end

# data structure to accomodate generalized parallel MCTS algorithm published here: https://arxiv.org/abs/2006.08785
struct MCTSPar{T <: MCTSAlgo, F <: Function, S, A} 
    𝒫::MDP # problem
    N::Vector{Dict{Tuple{S, A}, Int64}}  # visit counts
    Q::Vector{Dict{Tuple{S, A}, Float64}}  # action value estimates
    O::Vector{Dict{Tuple{S, A}, Int64}}  #on-going simulations 
    O_bar::Vector{Dict{Tuple{S, A}, Float64}} #average across rollouts of on-going simulations
    R::Vector{Dict{Tuple{S, A}, Vector{Tuple{Float64, Int64}}}} #tree sync stats
    d::Int64 # depth
    m::Int64 # number of simulations
    c::Float64 # exploration constant
    U::F # value function estimate
    F::Vector{Union{Task, Nothing}} # keeps track of each simulator whether it is occupied, ready to have results fetched, or unassigned
    M::Int64 # number of search trees
    algo::T
end



function init_MCTSPar(𝒫::MDP, rootstate::S, U::Func, numtrees::Integer, numsims::Integer, algo::T; d = 10, c = 10., m = 1000) where T <: MCTSAlgo where S where Func <: Function
    Action = eltype(𝒫.𝒜)
    N = [Dict{Tuple{S, Action}, Int64}() for _ in 1:numtrees]
    Q = [Dict{Tuple{S, Action}, Float64}() for _ in 1:numtrees]
    O = [Dict{Tuple{S, Action}, Int64}() for _ in 1:numtrees]
    O_bar = [Dict{Tuple{S, Action}, Float64}() for _ in 1:numtrees]
    R = [Dict{Tuple{S, Action}, Vector{Tuple{Float64, Int64}}}() for _ in 1:numtrees]
    F = Vector{Union{Task, Nothing}}(undef, numsims)
    fill!(F, nothing)
    MCTSPar{T, Func, S, Action}(𝒫, N, Q, O, O_bar, R, d, m, c, U, F, numtrees, algo)
end



struct UCT <: MCTSAlgo end 
struct LeafP <: MCTSAlgo end
struct RootP <: MCTSAlgo end
struct TreeP <: MCTSAlgo end 
struct WU_UCT <: MCTSAlgo end
struct VL_UCT_hard <: MCTSAlgo
    r #virtual loss
end
struct VL_UCT_soft <: MCTSAlgo
    r #virtual loss
    n #soft adjustment parameter
end
struct BU_UCT <: MCTSAlgo
    m_max #∈(0,1)
end

calc_τ_syn(::MCTSPar) = 1
calc_τ_syn(pol::MCTSPar{LeafP}) = pol.M 
calc_τ_syn(pol::MCTSPar{RootP}) = pol.m

f_sel(pol::MCTSPar, m1, m2) = 1
f_sel(pol::MCTSPar{LeafP}, m1, m2) = (m1 + 1) % pol.M 
f_sel(pol::MCTSPar{RootP}, m1, m2) = m1
f_sel(pol::MCTSPar{T}, m1, m2) where T <: Union{TreeP, WU_UCT, VL_UCT_hard, VL_UCT_soft} = rand(1:pol.M)

calcQ̃(pol::MCTSPar, s_a, m) = 0.0
calcQ̃(pol::MCTSPar{T}, s_a, m) where T <: Union{VL_UCT_hard, VL_UCT_soft} = -pol.algo.r 
calcQ̃(pol::MCTSPar{BU_UCT}, s_a, m) = 1.0 

calcÑ(pol::MCTSPar, s_a, m) = 0
calcÑ(pol::MCTSPar{T}, s_a, m) where T <: Union{WU_UCT, BU_UCT} = pol.O[m][s_a]
calcÑ(pol::MCTSPar{VL_UCT_soft}, s_a, m) = pol.algo.n*pol.O[m][s_a]

α(pol::MCTSPar, s_a, m) = 1. 
α(pol::MCTSPar{VL_UCT_soft}, s_a, m) = pol.N[m][s_a] / (pol.N[m][s_a] + pol.algo.n*pol.O[m][s_a]) 

β(pol::MCTSPar, s_a, m) = 0.
β(pol::MCTSPar{VL_UCT_hard}, s_a, m) = pol.O[m][s_a] 
β(pol::MCTSPar{VL_UCT_soft}, s_a, m) = pol.algo.n*pol.O[m][s_a] / (pol.N[m][s_a] + pol.algo.n*pol.O[m][s_a])
β(pol::MCTSPar{BU_UCT}, s_a, m) = pol.O_bar[m][s_a] < pol.algo.m_max*length(pol.F) ? 0. : -Inf 

calcQ̄(pol::MCTSPar, s_a, m) = α(pol, s_a, m)*pol.Q[m][s_a] + β(pol, s_a, m)*calcQ̃(pol, s_a, m) 
calcN̄(pol::MCTSPar, s_a, m) = pol.N[m][s_a] + calcÑ(pol, s_a, m) 

function (π::MonteCarloTreeSearch)(s)
    for k in 1:π.m
        simulate!(π, s)
    end
    return argmax(a->π.Q[(s,a)], π.𝒫.𝒜)
end

function (π::MCTSPar)(s, clear_dicts=true)
    mcts_rollout!(π::MCTSPar, s)
    Q = synctrees!(π)
    action = argmax(a->Q[(s,a)], π.𝒫.𝒜)
    clear_dicts && clear_dicts!(π)
    return action
end 

function simulate!(π::MonteCarloTreeSearch, s, d=π.d)
    if d ≤ 0
        return π.U(s)
    end
    𝒫, N, Q, c, O = π.𝒫, π.N, π.Q, π.c, π.O
    𝒜, TR, γ = 𝒫.𝒜, 𝒫.TR, 𝒫.γ
    if !haskey(N, (s, first(𝒜)))
        for a in 𝒜
            N[(s,a)] = 0
            Q[(s,a)] = 0.0
        end
        return π.U(s)
    end
    a = explore(π, s)
    s′, r = TR(s,a)
    q = r + γ*simulate!(π, s′, d-1)
    N[(s,a)] += 1
    Q[(s,a)] += (q-Q[(s,a)])/N[(s,a)]
    return q
end

@tailrec function mcts_rollout!(π::MCTSPar, s0, rolloutnum = 1, finishedsims = 0, initiatedsims = 0, m1 = π.M, m2 = 1)
    #once all simulation tasks are completed, end rollout
    # (unfinishedsims == 0) && return nothing
    (finishedsims == π.m) && return nothing

    
    (U, F) = (π.U, π.F)

    # println("$finishedsims simulations have been completed.  $(count(a -> isnothing(a) || istaskdone(a), F)) simulators are idle")
    
    #only do a new rollout if there are simulations left to queue up
    if initiatedsims < π.m
        #m1 and m2 represent the index of the search tree previously selected and previously updated with backpropagation respectively
        treeindex = f_sel(π, m1, m2)
        
        #starting from s0 recursively traverse tree using exploration function until reaching a state which has never been visited before
        #also update the ongoing simulation count for every (s,a) pair visited
        (s, traj) = node_selection!(π, s0, rolloutnum, treeindex)
        
        #this will result in a crash if no element in F is nothing.  this situation should not happen though because if all simulators are occupied then the rollout will not proceed until one is freed
        selectedsim = findfirst(isnothing, F)
        
        #begin a simulation task and store the future result in F
        F[selectedsim] = @tspawnat (selectedsim + 1) (U(s), treeindex, traj) 
        initiatedsims += 1
        rolloutnum += 1
    else
        treeindex = m1
    end
    
    #check to see if any simulators are unoccupied
    if mapreduce(isnothing, (a,b) -> a || b, F) && (initiatedsims < π.m)
        #if there are still unoccupied simulators and still simulations to initiate, then restart the procedure updated the selected tree index
        mcts_rollout!(π, s0, rolloutnum, finishedsims, initiatedsims, treeindex, m2)
    else 
        #otherwise, wait for a result and complete backpropagation and tree sync
        getsim() = findfirst(a -> !isnothing(a) && istaskdone(a), F)
        while isnothing(getsim())
        end
        fetchindex = getsim()
        
        (qterm, simtree, simtraj) = fetch(F[fetchindex]) 
        
        backprop!(π, simtree, simtraj, qterm)

        #after finishing backprop, reset this simulator to idle
        F[fetchindex] = nothing
        finishedsims += 1
        (finishedsims % calc_τ_syn(π) == 0) && synctrees!(π)

        mcts_rollout!(π, s0, rolloutnum, finishedsims, initiatedsims, treeindex, simtree)
    end
end

function resetN!(π::MCTSPar{BU_UCT}, N, trajectory)
    if length(trajectory) >= 2
        (s,a,r) = trajectory[end-1]
        N[(s,a)] = 1
    end
    return nothing
end

resetN!(π, N, trajectory) = return nothing

@tailrec function node_selection!(π::MCTSPar, s, n, m, d=π.d, trajectory=Vector{Tuple}())
    (d == 0) && return (s, trajectory)
    𝒫, N, O, O_bar, Q, c = π.𝒫, π.N[m], π.O[m], π.O_bar[m], π.Q[m], π.c
    𝒜, TR = 𝒫.𝒜, 𝒫.TR
    if !haskey(N, (s, first(𝒜)))
    #in this case we've reached a new state in the tree
        for a in 𝒜
            N[(s,a)] = 0
            Q[(s,a)] = 0.0
            O[(s,a)] = 0
            O_bar[(s,a)] = 0
        end
         #for BU_UCT only, this function will reset the visit count of the state 2 visits prior to the new state in order to treat all previous simulation results as 1
         resetN!(π, N, trajectory)
         return (s, trajectory)
    end

    #if we aren't at maximum depth or a new state, then select a new action with the exploration formula
    # Ns = foldl(+, Map(a -> calcN̄(π, (s, a), m)), 𝒜)
    Ns = sum(calcN̄(π, (s, a), m) for a in 𝒜)

    a = argmax(a-> calcQ̄(π, (s,a), m) + c*bonus(π, Ns, s, a, m), 𝒜)
    # a = rand(𝒜)

    (s_new, r) = TR(s, a)

    #update the ongoing simulation count for this (s,a) pair
     O[(s,a)] += 1
     #update running average of ongoing simulation count per rollout
     O_bar[(s,a)] += ((n-1)*O_bar[(s,a)] + O[(s,a)])/n 

    #continue traversing through the tree
    node_selection!(π, s_new, n, m, d - 1, push!(trajectory, (s,a,r)))
end

function explore(π::MonteCarloTreeSearch, s)
    𝒜, N, Q, c = π.𝒫.𝒜, π.N, π.Q, π.c
    Ns = sum(N[(s,a)] for a in 𝒜)
    return argmax(a->Q[(s,a)] + c*bonus(N[(s,a)], Ns), 𝒜)
end

function bonus(π::MCTSPar, Ns, s, a, m)
    d = calcN̄(π, (s, a), m)
    d == 0 ? Inf : sqrt(2*log(Ns) / d)
end

bonus(Nsa, Ns) = Nsa == 0 ? Inf : sqrt(log(Ns)/Nsa)

@tailrec function backprop!(π::MCTSPar, m, traj, v_next, i = length(traj))
    (i == 0) && return nothing
    (𝒫, N, Q, O, R) = (π.𝒫, π.N[m], π.Q[m], π.O[m], π.R[m])
    γ = 𝒫.γ
    
    #get the action value pairs starting at the end of the trajectory
    (s,a,r) = traj[i]

    #update visit and simulation counts
    O[(s,a)] -= 1
    N[(s,a)] += 1
    n = N[(s,a)]

    #calculate discounted reward starting from the simluation value estimate
    v = r + γ*v_next

    #update stored statistics for tree sync
    if haskey(R, (s,a))
        push!(R[(s,a)], (v_next, 0))
    else 
        R[(s,a)] = [(v_next, 0)]
    end

    #update Q values
    Q[(s,a)] = Q[(s,a)]*((n - 1)/n) + (v / n)
    
    #proceed to the previous spot on the trajectory
    backprop!(π, m, traj, v, i-1)
end

#for any algo where the tree sync interval is 1, it is effectively the same as only having one search tree, so syncing is not necessary
function synctrees!(π::MCTSPar{T}) where T <: Union{UCT, TreeP, WU_UCT, VL_UCT_hard, VL_UCT_soft, BU_UCT}
    return π.Q[1]
end

function synctrees!(π)
    O = reduce(combine_dicts(+), π.O)
    replace_dicts!(π.O, O)
    O_bar = reduce(combine_dicts(+), π.O_bar)
    replace_dicts!(π.O_bar, O_bar)

    #only select elements from the first tree that have been previously synchronized
    R = Dict(k => filter(a -> a[2] == 1, π.R[1][k]) for k in keys(π.R[1]))

    for Ri in π.R
        for k in keys(Ri)
            for (r, psi) in Ri[k] 
                if psi == 0
                    if haskey(R, k)
                        push!(R[k], (r, 1))
                    else
                        R[k] = [(r, 1)]
                    end
                end
            end
        end
    end

    #i think this is necessary to update the synchronization indicator but it isn't listed in the appendix
    replace_dicts!(π.R, R)

    Q = Dict(k => sum(v for (v, psi) in R[k])/length(R[k]) for k in keys(R))
    N = Dict(k => length(R[k]) for k in keys(R))

    replace_dicts!(π.Q, Q)
    replace_dicts!(π.N, N)
    return Q
end 

function combine_dicts(op::Function, d1::T, d2::T) where T <: Dict
   dout = T() 
   klist = union(keys(d1), keys(d2))
   for k in klist 
        if haskey(d1, k) && haskey(d2, k)
            dout[k] = op(d1[k], d2[k])
        elseif haskey(d1, k)
            dout[k] = d1[k]
        else # elseif(haskey(d2, k))
            dout[k] = d2[k]
        end
   end
   return dout
end

combine_dicts(op::Function) = (d1,d2) -> combine_dicts(op, d1, d2)

function replace_dicts!(dlist::Vector{T}, dnew::T) where T <: Dict
    for i in eachindex(dlist)
        dlist[i] = dnew
    end
end

function clear_dicts!(π::MCTSPar)
    for dlist in (π.N, π.Q, π.O, π.O_bar, π.R)
        for i in eachindex(dlist)
            dlist[i] = typeof(dlist[i])()
        end
    end 
end

end # module
