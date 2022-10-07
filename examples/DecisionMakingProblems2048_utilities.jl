using MCTSExperiments
using BenchmarkTools
using Statistics
using StatsBase
using DecisionMakingProblems
using OnlineStats
using Transducers
using DataFrames

#some convenience names
import DecisionMakingProblems.Board
Action = DecisionMakingProblems.TwentyFortyEightAction
import DecisionMakingProblems.initial_board
import DecisionMakingProblems.print_board
import DecisionMakingProblems.score_board
import DecisionMakingProblems.move
import DecisionMakingProblems.insert_tile_rand
import DecisionMakingProblems.draw_tile
import DecisionMakingProblems.isdone
import DecisionMakingProblems.DIRECTIONS
import DecisionMakingProblems.get_max_rank
#=
const LEFT = 0x00
const DOWN = 0x01
const RIGHT = 0x02
const UP = 0x03
=#

import DecisionMakingProblems.transition_and_reward

export print_board, initial_board, mdp_2048, random_2048_policy, run_random_2048_rollout, play_game, getgamestats

function transition_and_reward(::TwentyFortyEight, s::Board, a::Action)
    if isdone(s)
        return (s, -1.0f0)
    end
    s′ = move(s, a)
    if s′ == s # illegal action
        return (s′, -1.0f0)
    end
    s′ = insert_tile_rand(s′, draw_tile())
    r = score_board(s′) - score_board(s)
    return (s′, r)
end

const twenty_forty_eight = DecisionMakingProblems.TwentyFortyEight(γ=0.99)
const mdp_2048 = DecisionMakingProblems.MDP(twenty_forty_eight)
#=
mdp_2048 is the MDP defining the game with discount factor, action space, reward function, and sample transition and reward defined.  The state space and transition function are not defined.
julia> dump(mdp_2048)
MDP
  γ: Float64 0.99
  𝒮 : Nothing nothing
  𝒜 : NTuple{4, UInt8}
    4: UInt8 0x03
  T: Nothing nothing
  R: #27 (function of type DecisionMakingProblems.var"#27#29"{DecisionMakingProblems.TwentyFortyEight})
    mdp: DecisionMakingProblems.TwentyFortyEight
      γ: Float64 0.99
  TR: #28 (function of type DecisionMakingProblems.var"#28#30"{DecisionMakingProblems.TwentyFortyEight})
    mdp: DecisionMakingProblems.TwentyFortyEight
      γ: Float64 0.99
=#

#create a random policy that selects moves at random from the available 4 directions
random_2048_policy(board::Board) = rand(DIRECTIONS)

#returns the future disconted reward for rolling out the policy for the number of steps as specified by d
run_random_2048_rollout(board::Board, d::Integer) = rollout(mdp_2048.TR, mdp_2048.γ, board, random_2048_policy, d, isdone)
run_random_2048_rollout(board::Board) = rollout(mdp_2048.TR, mdp_2048.γ, board, random_2048_policy, isdone)

#now let's create some MCTS policies for 2048 that differ by their value function estimate

#with this function we can initialize a policy with empty dictionaries
function create_mcts_policy(U::Function; d = 10, m = 100, c = 100.0)
    MCTSExperiments.MonteCarloTreeSearch(
        mdp_2048, # 𝒫, MDP problem 
        Dict{Tuple{Board, Action}, Int64}(), # N, visit counts for each state/action pair
        Dict{Tuple{Board, Action}, Float32}(), # Q, action value estimates for each state/action pair
        d, # maximum depth = 10 by default
        m, # number of simulations = 100 by default
        c, # exploration constant = 100 by default
        U # value function estimate 
    )
end

# Copied the following function to play a game from DecisionMakingProblems but modified it to return the final score and not print anything.  Added illegal move maximum to prevent policies from continuing to attempt illegal moves forever.
 """
Play 2048 to completion using the given policy.
The final score is returned.
Note that this core is "correct" in that we track whether 2 or 4 tiles are generated
and update the score appropriately.
"""
function play_game(π::Function; max_illegal = 10000)
    s = initial_board()

    # Number of moves.
    moveno = 0

    # Number of illegal moves.
    num_illegal = 0

    # Cumulative penalty for obtaining free 4 tiles, as
    # when computing the score of merged tiles we cannot distinguish between
    # merged 2-tiles and spawned 4 tiles.
    scorepenalty = score_board(s)

    while !DecisionMakingProblems.isdone(s) && num_illegal < max_illegal

        moveno += 1
        # println("Move #$(moveno), current score=$(score_board(s) - scorepenalty)")
        # print_board(s)

        a = π(s)
        if a == DecisionMakingProblems.NONE
            break
        end

        # println("\ta = ", DecisionMakingProblems.TWENTY_FORTY_EIGHT_MOVE_STRINGS[a+1])

        s′ = DecisionMakingProblems.move(s, a)
        if s′ == s
            # @warn "Illegal move!"
            moveno -= 1
            num_illegal += 1
            continue
        else
            num_illegal = 0
        end

        tile = DecisionMakingProblems.draw_tile()
        if tile == 2
            scorepenalty += 4
        end
        s = DecisionMakingProblems.insert_tile_rand(s′, tile)
        println("On move $moveno with a rank of $(get_max_rank(s))") 
        print_board(s)
    end
    return score_board(s) - scorepenalty, DecisionMakingProblems.get_max_rank(s), moveno
end

getgamestats(games) = map(i -> summarystats([a[i] for a in games]), 1:3)