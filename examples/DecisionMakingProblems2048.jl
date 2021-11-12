using MCTS
using BenchmarkTools
using Statistics
using StatsBase
using DecisionMakingProblems

#some convenience names
Board = DecisionMakingProblems.Board
Action = DecisionMakingProblems.TwentyFortyEightAction
initial_board = DecisionMakingProblems.initial_board
print_board = DecisionMakingProblems.print_board
score_board = DecisionMakingProblems.score_board
#=
const LEFT = 0x00
const DOWN = 0x01
const RIGHT = 0x02
const UP = 0x03
=#

twenty_forty_eight = DecisionMakingProblems.TwentyFortyEight(γ=0.99)
mdp_2048 = DecisionMakingProblems.MDP(twenty_forty_eight)

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

#create a starting board
init_board = initial_board()
#=
julia> DecisionMakingProblems.print_board(init_board)
     0     0     2     0
     0     0     0     0
     0     2     0     0
     0     0     0     0
=#
#create a random policy that selects moves at random from the available 4 directions
random_2048_policy(board::Board) = rand(DecisionMakingProblems.DIRECTIONS)

#returns the future disconted reward for rolling out the policy for the number of steps as specified by d
run_random_2048_rollout(d::Integer, board::Board) = MCTS.rollout(mdp_2048, board, random_2048_policy, d)

#generate statistics on the random policy playing for 100 moves
random_2048_rollouts = [run_random_2048_rollout(100, init_board) for _ in 1:10000]
summarystats(random_2048_rollouts)
#=
Summary Stats:
Length:         10000
Missing Count:  0
Mean:           455.526485
Minimum:        53.318648
1st Quartile:   404.462007
Median:         457.705996
3rd Quartile:   525.514537
Maximum:        690.029171
=#

#now let's create some MCTS policies for 2048 that differ by their value function estimate

#with this function we can initialize a policy with empty dictionaries
function create_mcts_policy(U::Function; d = 10, m = 100, c = 100.0)
    MCTS.MonteCarloTreeSearch(
        mdp_2048, # 𝒫, MDP problem 
        Dict{Tuple{Board, Action}, Int64}(), # N, visit counts for each state/action pair
        Dict{Tuple{Board, Action}, Float32}(), # Q, action value estimates for each state/action pair
        d, # maximum depth = 10 by default
        m, # number of simulations = 100 by default
        c, # exploration constant = 100 by default
        U # value function estimate 
    )
end

#the simplest mcts policy will just use the reward of the current board state as the value function estimate
mcts_board_score = create_mcts_policy(s -> DecisionMakingProblems.score_board(s))

#we can directly use the policy to evaluate the move it would suggest for the initial board
mcts_board_score(init_board) #0x03 which corresponds to UP

#=after running this evaluation the N and Q dictionaries are now populated
julia> mcts_board_score.N
Dict{Tuple{UInt64, UInt8}, Int64} with 388 entries:
  (0x0000000010000210, 0x03) => 0
  (0x0110000000100000, 0x00) => 1
  (0x2100010000000000, 0x00) => 0
  (0x0110000000000020, 0x01) => 0
  (0x0001000100000001, 0x00) => 9
  (0x0000100000001001, 0x00) => 0
  (0x0000000000100110, 0x00) => 0
  (0x1110000001000000, 0x00) => 0
  ⋮                          => ⋮

julia> mcts_board_score.Q
Dict{Tuple{UInt64, UInt8}, Float64} with 388 entries:
  (0x0000000010000210, 0x03) => 0.0
  (0x0110000000100000, 0x00) => 7.96
  (0x2100010000000000, 0x00) => 0.0
  (0x0110000000000020, 0x01) => 0.0
  (0x0001000100000001, 0x00) => -4.86897
  (0x0000100000001001, 0x00) => 0.0
  (0x0000000000100110, 0x00) => 0.0
  (0x1110000001000000, 0x00) => 0.0
  ⋮                          => ⋮
=#

#we can also benchmark this single evaluation
@benchmark mcts_board_score(init_board) 
#=
BenchmarkTools.Trial: 1142 samples with 1 evaluation.
 Range (min … max):  4.026 ms …  11.561 ms  ┊ GC (min … max): 0.00% … 58.64%
 Time  (median):     4.191 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   4.371 ms ± 788.721 μs  ┊ GC (mean ± σ):  1.86% ±  6.54%

  ▁█
  ██▄▂▃▅▃▂▂▂▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂ ▂
  4.03 ms         Histogram: frequency by time        10.9 ms <

 Memory estimate: 969.38 KiB, allocs estimate: 39298.
=#

#we can explore how adjusting the number of simulations
@benchmark create_mcts_policy(s -> DecisionMakingProblems.score_board(s), m = 1000)(init_board)
#=
BenchmarkTools.Trial: 380 samples with 1 evaluation.
 Range (min … max):  12.147 ms … 19.886 ms  ┊ GC (min … max): 0.00% … 32.41%
 Time  (median):     12.816 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   13.152 ms ±  1.384 ms  ┊ GC (mean ± σ):  2.29% ±  6.83%

     ▁▁█▆▁
  ▅▇██████▅▃▄▄▄▂▃▂▁▃▂▁▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▂▂▂▃▂▁▃ ▃
  12.1 ms         Histogram: frequency by time        18.9 ms <

 Memory estimate: 4.16 MiB, allocs estimate: 126453.
 =#
@benchmark create_mcts_policy(s -> DecisionMakingProblems.score_board(s), m = 10000)(init_board)
#=
BenchmarkTools.Trial: 26 samples with 1 evaluation.
 Range (min … max):  186.760 ms … 207.079 ms  ┊ GC (min … max): 0.00% … 2.65%
 Time  (median):     198.741 ms               ┊ GC (median):    2.90%
 Time  (mean ± σ):   197.997 ms ±   5.298 ms  ┊ GC (mean ± σ):  1.98% ± 1.47%

        ▃                       ▃  ▃       █▃
  ▇▁▁▁▇▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▇▁▁▁▁▁▇▁▁▇█▇▁█▁▇▇▇▇▁▇██▁▁▁▁▁▇▁▁▁▁▁▁▁▇▁▁▇▇ ▁
  187 ms           Histogram: frequency by time          207 ms <

 Memory estimate: 51.68 MiB, allocs estimate: 1951557.

 So at 10000 simulations we see garbage collection having a noticeable effect on the runtime

 =#


#now we can time the MCTS policy evaluation on a different version of the policy that uses a 10 step random rollout for the value function estimate 
mcts_rollout = create_mcts_policy(s -> run_random_2048_rollout(10, s))
mcts_rollout(init_board)
@benchmark mcts_rollout(init_board)
#=
BenchmarkTools.Trial: 964 samples with 1 evaluation.
 Range (min … max):  4.500 ms … 16.948 ms  ┊ GC (min … max): 0.00% … 65.19%
 Time  (median):     4.892 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   5.175 ms ±  1.218 ms  ┊ GC (mean ± σ):  2.31% ±  7.02%

   █
  ▆█▇▄▇█▃▂▂▂▂▁▁▁▁▁▁▁▂▁▁▁▁▁▁▁▁▁▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂ ▂
  4.5 ms         Histogram: frequency by time        15.4 ms <

 Memory estimate: 1.15 MiB, allocs estimate: 47716.
=#

#not a huge difference in time for the default 100 simulations, but let's see how it scales up
@benchmark create_mcts_policy(s -> run_random_2048_rollout(10, s), m = 1000)(init_board)
#=
BenchmarkTools.Trial: 224 samples with 1 evaluation.
 Range (min … max):  20.230 ms … 33.433 ms  ┊ GC (min … max): 0.00% … 28.86%
 Time  (median):     20.796 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   22.338 ms ±  2.990 ms  ┊ GC (mean ± σ):  2.79% ±  7.19%

  ▅█▇▅▂               ▂▂▁
  █████▆▁▄▄▁▁▄▆▇▆▇▄██▇███▆▁▁▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▇▆▁▁▁▁▁▆▆▄▁▄▁▄▁▆ ▆
  20.2 ms      Histogram: log(frequency) by time      32.9 ms <

 Memory estimate: 7.57 MiB, allocs estimate: 268809.
=#

@benchmark create_mcts_policy(s -> run_random_2048_rollout(10, s), m = 10000)(init_board)
#=
BenchmarkTools.Trial: 17 samples with 1 evaluation.
 Range (min … max):  280.152 ms … 304.394 ms  ┊ GC (min … max): 0.00% … 3.36%
 Time  (median):     298.187 ms               ┊ GC (median):    3.34%
 Time  (mean ± σ):   295.943 ms ±   6.804 ms  ┊ GC (mean ± σ):  2.59% ± 1.47%

  ▁          ▁        ▁      ▁ ▁    █        ▁ █     █▁▁▁ ▁   ▁  
  █▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁█▁▁▁▁▁▁█▁█▁▁▁▁█▁▁▁▁▁▁▁▁█▁█▁▁▁▁▁████▁█▁▁▁█ ▁
  280 ms           Histogram: frequency by time          304 ms <

 Memory estimate: 84.80 MiB, allocs estimate: 3337260.
 =#

 #so this rollout method is 1.5-2x slower if the number of simulations scale up, but what is the impact on how well it plays the game?  Copied the following function to play a game from DecisionMakingProblems but modified it to return the final score and not print anything.  Added illegal move maximum to prevent policies from continuing to attempt illegal moves forever.

 """
Play 2048 to completion using the given policy.
The final score is returned.
Note that this core is "correct" in that we track whether 2 or 4 tiles are generated
and update the score appropriately.
"""
function play_game(π::Function; max_illegal = 10)
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
    end
    return score_board(s) - scorepenalty, DecisionMakingProblems.get_max_rank(s), moveno
end

#we can use this to play a game to completion for a given policy.  Let's try it with the random policy
(random_score, random_rank, random_move_count) = play_game(random_2048_policy)

#using this function we can get statistics about games played with the random policy
random_games = [play_game(random_2048_policy) for _ in 1:1000]
summarystats([a[1] for a in random_games])
#=
Summary Stats:
Length:         1000
Missing Count:  0
Mean:           980.591980
Minimum:        164.000000
1st Quartile:   608.000000
Median:         948.000000
3rd Quartile:   1268.000000
Maximum:        3076.000000
=#
summarystats([a[2] for a in random_games])
#=
Summary Stats:
Length:         1000
Missing Count:  0
Mean:           6.450000
Minimum:        4.000000
1st Quartile:   6.000000
Median:         6.000000
3rd Quartile:   7.000000
Maximum:        8.000000
=#

summarystats([a[3] for a in random_games])
#=
Summary Stats:
Length:         1000
Missing Count:  0
Mean:           103.605000
Minimum:        35.000000
1st Quartile:   79.000000
Median:         101.000000
3rd Quartile:   125.000000
Maximum:        247.000000
=#

#so now we have some baseline stats to work with for all the arelevent game characteristics

#also we can benchmark the play game function to see how long the random policy takes
@benchmark play_game(random_2048_policy)
#=
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):   5.300 μs … 65.000 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     19.200 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   20.027 μs ±  6.840 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

             ▂▄▇▇█▇▇█▇▆▆▇▆▆▆▅▆▆▆▃▄▃▂▁
  ▁▁▂▂▂▄▄▆▆▇█████████████████████████▇▆▆▆▄▄▅▄▃▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂ ▅
  5.3 μs          Histogram: frequency by time        40.6 μs <

 Memory estimate: 1.17 KiB, allocs estimate: 74.
 =#

 #now let's try playing a game with the MCTS policy that uses the board score as the value function estimate and uses 100 simulations.
mcts_score_100_policy = create_mcts_policy(s -> score_board(s))
mcts_score_100_game = play_game(s -> mcts_score_100_policy(s))

#=
julia> @benchmark play_game(s -> mcts_score_100_policy(s))
BenchmarkTools.Trial: 7 samples with 1 evaluation.
 Range (min … max):  453.885 ms …    1.333 s  ┊ GC (min … max): 0.00% … 3.05%
 Time  (median):     872.985 ms               ┊ GC (median):    3.47%
 Time  (mean ± σ):   868.886 ms ± 277.910 ms  ┊ GC (mean ± σ):  2.41% ± 1.84%

  ▁               ▁     ▁      █            ▁                 ▁
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁█▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  454 ms           Histogram: frequency by time          1.33 s <

 Memory estimate: 91.40 MiB, allocs estimate: 3722578
 =#

mcts_score_100_games = [play_game(b -> create_mcts_policy(s -> score_board(s))(b)) for _ in 1:100]
summarystats([a[1] for a in mcts_score_100_games])
#= 
Summary Stats:
Length:         100
Missing Count:  0
Mean:           7405.680176
Minimum:        3012.000000
1st Quartile:   5795.000000
Median:         6976.000000
3rd Quartile:   7665.000000
Maximum:        15344.000000
=#
summarystats([a[2] for a in mcts_score_100_games])
#= 
Summary Stats:
Length:         100
Missing Count:  0
Mean:           9.130000
Minimum:        8.000000
1st Quartile:   9.000000
Median:         9.000000
3rd Quartile:   9.000000
Maximum:        10.000000
=#
summarystats([a[3] for a in mcts_score_100_games])
#= 
Summary Stats:
Length:         100
Missing Count:  0
Mean:           456.810000
Minimum:        238.000000
1st Quartile:   380.000000
Median:         451.000000
3rd Quartile:   488.000000
Maximum:        830.000000
=#

#now let's compare to an mcts policy that uses a 10 step random rollout instead of the board score for the value function estimate
mcts_rollout_100_policy = create_mcts_policy(s -> run_random_2048_rollout(10, s))
mcts_rollout_100_game = play_game(s -> mcts_rollout_100_policy(s)) #(35704, 11, 1741)
mcts_rollout_100_games = [play_game(b -> create_mcts_policy(s -> run_random_2048_rollout(10, s))(b)) for _ in 1:100]

#even with 100 simulations this seems to give much better results, with just the first game already reaching the max rank.  What if we do it with 1000 simulations instead
mcts_rollout_1000_policy = create_mcts_policy(s -> run_random_2048_rollout(10, s), m = 1000)
mcts_rollout_1000_game = play_game(s -> mcts_rollout_1000_policy(s))  #(35716.0f0, 11, 1743)
#=
julia> summarystats([a[1] for a in mcts_rollout_100_games])
Summary Stats:
Length:         100
Missing Count:  0
Mean:           24562.320312
Minimum:        6864.000000
1st Quartile:   15848.000000
Median:         26638.000000
3rd Quartile:   32474.000000
Maximum:        54076.000000


julia> summarystats([a[2] for a in mcts_rollout_100_games])
Summary Stats:
Length:         100
Missing Count:  0
Mean:           10.570000
Minimum:        9.000000
1st Quartile:   10.000000
Median:         11.000000
3rd Quartile:   11.000000
Maximum:        12.000000


julia> summarystats([a[3] for a in mcts_rollout_100_games])
Summary Stats:
Length:         100
Missing Count:  0
Mean:           1221.290000
Minimum:        429.000000
1st Quartile:   872.500000
Median:         1289.000000
3rd Quartile:   1535.250000
Maximum:        2284.000000
=#

#and with 10000
# mcts_rollout_10000_policy = create_mcts_policy(s -> run_random_2048_rollout(10, s), m = 10000)
# mcts_rollout_10000_game = play_game(s -> mcts_rollout_10000_policy(s))  #(35716.0f0, 11, 1743)




#=
julia> @benchmark play_game(s -> mcts_score_100_policy(s))
BenchmarkTools.Trial: 7 samples with 1 evaluation.
 Range (min … max):  453.885 ms …    1.333 s  ┊ GC (min … max): 0.00% … 3.05%
 Time  (median):     872.985 ms               ┊ GC (median):    3.47%
 Time  (mean ± σ):   868.886 ms ± 277.910 ms  ┊ GC (mean ± σ):  2.41% ± 1.84%

  ▁               ▁     ▁      █            ▁                 ▁
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁█▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  454 ms           Histogram: frequency by time          1.33 s <

 Memory estimate: 91.40 MiB, allocs estimate: 3722578
 =#

 #What if we only use 10 simulations
mcts_rollout_10_policy = create_mcts_policy(s -> run_random_2048_rollout(10, s), m = 10)
mcts_rollout_10_game = play_game(s -> mcts_rollout_10_policy(s)) #(6272.0f0, 9, 402)
@benchmark play_game(s -> mcts_rollout_10_policy(s)) 
#=
julia> @benchmark play_game(s -> mcts_rollout_10_policy(s))
BenchmarkTools.Trial: 60 samples with 1 evaluation.
 Range (min … max):   5.135 ms … 204.774 ms  ┊ GC (min … max): 0.00% … 4.08%
 Time  (median):     82.556 ms               ┊ GC (median):    5.45%
 Time  (mean ± σ):   84.444 ms ±  45.518 ms  ┊ GC (mean ± σ):  5.86% ± 9.70%

         ▃    ▃▃█ ▃█      ▃█  ▃██                ▃    ▃
  ▇▁▇▁▁▇▇█▁▇▇▁███▇██▇▁▁▇▇▁██▁▇███▇▇▇▁▁▇▁▇▁▇▇▁▇▇▁▇█▇▇▇▁█▇▇▇▁▁▁▇ ▁
  5.14 ms         Histogram: frequency by time          170 ms <

 Memory estimate: 1.73 MiB, allocs estimate: 71121.
=#
mcts_rollout_10_games = [play_game(b -> create_mcts_policy(s -> run_random_2048_rollout(10, s), m = 10)(b)) for _ in 1:100]
#=
julia> summarystats([a[1] for a in mcts_rollout_10_games])
Summary Stats:
Length:         100
Missing Count:  0
Mean:           8619.280273
Minimum:        1368.000000
1st Quartile:   6200.000000
Median:         6958.000000
3rd Quartile:   12002.000000
Maximum:        15868.000000


julia> summarystats([a[2] for a in mcts_rollout_10_games])
Summary Stats:
Length:         100
Missing Count:  0
Mean:           9.240000
Minimum:        7.000000
1st Quartile:   9.000000
Median:         9.000000
3rd Quartile:   10.000000
Maximum:        10.000000


julia> summarystats([a[3] for a in mcts_rollout_10_games])
Summary Stats:
Length:         100
Missing Count:  0
Mean:           511.090000
Minimum:        131.000000
1st Quartile:   394.750000
Median:         452.500000
3rd Quartile:   675.000000
Maximum:        878.000000
=#

#once again let's compare to the random policy playing a full game to completion
random_game = play_game(random_2048_policy) #(6272.0f0, 9, 402)
@benchmark play_game(random_2048_policy) 
#=
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):   1.100 μs …  2.417 ms  ┊ GC (min … max): 0.00% … 98.49%
 Time  (median):     15.200 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   16.396 μs ± 24.848 μs  ┊ GC (mean ± σ):  1.45% ±  0.98%

              ▁▂▄▆█▇█▃██▇▇▇▇▇▅▅▃▃▁▂
  ▁▁▁▁▁▁▁▂▃▄▅▇█████████████████████▇▇▆▅▅▅▃▄▃▄▃▃▃▃▃▂▃▂▂▂▂▂▁▂▂▁ ▄
  1.1 μs          Histogram: frequency by time        36.2 μs <

 Memory estimate: 80 bytes, allocs estimate: 4.
=#
random_games = [play_game(random_2048_policy) for _ in 1:10000]
random_game_stats = map(i -> summarystats([a[i] for a in random_games]), 1:3)
#=
3-element Vector{StatsBase.SummaryStats}:
 Summary Stats:
Length:         10000
Missing Count:  0
Mean:           999.621582
Minimum:        0.000000
1st Quartile:   616.000000
Median:         948.000000
3rd Quartile:   1284.000000
Maximum:        3408.000000

 Summary Stats:
Length:         10000
Missing Count:  0
Mean:           6.477400
Minimum:        2.000000
1st Quartile:   6.000000
Median:         7.000000
3rd Quartile:   7.000000
Maximum:        8.000000

 Summary Stats:
Length:         10000
Missing Count:  0
Mean:           104.739000
Minimum:        0.000000
1st Quartile:   80.000000
Median:         101.000000
3rd Quartile:   126.000000
Maximum:        265.000000
=#
