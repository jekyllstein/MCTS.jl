using Pkg 
Pkg.activate(@__DIR__)
include("DecisionMakingProblems2048_utilities.jl")
using Base.Threads

#create a starting board
init_board = initial_board()
#=
julia> DecisionMakingProblems.print_board(init_board)
     0     0     2     0
     0     0     0     0
     0     2     0     0
     0     0     0     0
=#

#using the random policy function we can generate moves from the initial board state
random_2048_policy(init_board) #0x02 which corresponds to right

@btime random_2048_policy($init_board)
#=
5.900 ns (0 allocations: 0 bytes)
=#


function rollout_game_stats(input...; ntrials = 1000)
    xf = 1:ntrials |> Map() do _
        (score, steps, sfinal) = run_random_2048_rollout(input...)
        (score = score, steps = steps, sfinal = sfinal)
    end 
    (scoreavg = mean(a -> a[1], xf), moveavg = mean(a -> a[2], xf), medianrank = median(xf |> Map(a -> get_max_rank(a[3])) |> collect))
end
# function rollout_game_stats(;ntrials = 1000)
#     xf = 1:ntrials |> Map() do _
#         (score, steps, sfinal) = run_random_2048_rollout(init_board)
#         (score = score, steps = steps, sfinal = sfinal)
#     end 
#     (scoreavg = mean(a -> a[1], xf), moveavg = mean(a -> a[2], xf), medianrank = median(xf |> Map(a -> get_max_rank(a[3])) |> collect))
# end

#if we use this policy for a rollout of 10 moves then we can see the reward accumulated after that many moves.  This will be important later for use in a value function estimate.
rollout_game_stats(init_board, 10)

#let's see how this compares to letting a game run until the terminal state
rollout_game_stats(init_board)

(score, steps, sfinal) = run_random_2048_rollout(init_board)
print_board(sfinal)
#=
julia> print_board(sfinal)
     2     8    16     8
     4   128    32     2
    32    64     4     8
     2     4    16     2
=#

#also let's get an example board after 10 moves to use for future evaluation
(_, _, board10) = run_random_2048_rollout(init_board, 10)
#=
julia> print_board(board10)
     8     0     0     0
     8     0     0     0
     2     4     0     0
     2     0     2     0
=#



#let's also benchmark random rollouts to see how the time scales with length
# rollout_times = [(moves = 2^n, times = @benchmark run_random_2048_rollout($init_board, $(2^n))) for n in 0:7]
#=
8-element Vector{NamedTuple{(:moves, :times), Tuple{Int64, BenchmarkTools.Trial}}}:
 (moves = 1, times = 186.331 ns)
 (moves = 2, times = 237.292 ns)
 (moves = 4, times = 364.251 ns)
 (moves = 8, times = 752.542 ns)
 (moves = 16, times = 1.450 μs)
 (moves = 32, times = 2.878 μs)
 (moves = 64, times = 5.250 μs)
 (moves = 128, times = 3.100 μs)
 
 So we see the time roughly doubling each time the rollout length is doubled as expected
 =#

 #=
finally what is the time for a rollout to termination?
julia> @btime run_random_2048_rollout(init_board)
  3.300 μs (4 allocations: 176 bytes)
=#

function get_random_game_stats(π, ngames = 1000)
    1:ngames |> Map(_ -> play_game(π)) |> getgamestats
end

 #=
 statistics for score, rank, and number of moves
 julia> get_random_game_stats(random_2048_policy)
3-element Vector{StatsBase.SummaryStats}:
 Summary Stats:
Length:         1000
Missing Count:  0
Mean:           1040.500000
Minimum:        4.000000
1st Quartile:   636.000000
Median:         998.000000
3rd Quartile:   1308.000000
Maximum:        3028.000000

 Summary Stats:
Length:         1000
Missing Count:  0
Mean:           6.468000
Minimum:        4.000000
1st Quartile:   6.000000
Median:         7.000000
3rd Quartile:   7.000000
Maximum:        8.000000

 Summary Stats:
Length:         1000
Missing Count:  0
Mean:           105.342000
Minimum:        6.000000
1st Quartile:   79.000000
Median:         102.000000
3rd Quartile:   127.000000
Maximum:        248.000000
=#

#now we can initialize a new MCTS policy, single tree not in parallel with a depth of 10, exploration constant of 10, and 1000 simulations with a random rollout of depth 10
mcts_UCT_1 = init_MCTSPar(mdp_2048, init_board, s -> run_random_2048_rollout(s, 10)[1], 1, 1, UCT(), d = 100, c = 100., m = 1000)

#we can select a move with this policy by calling it on the initial board state
mcts_UCT_1(init_board, false)
#=
julia> @btime mcts_UCT_rollout10_1(init_board)
  35.934 ms (397581 allocations: 13.93 MiB)
=#

#inside of any MCTS rollout the tree traversal and node selection happen repeatedly
MCTSExperiments.node_selection!(mcts_UCT_1, init_board, 1, 1, 10)
#=
julia> @btime MCTSExperiments.node_selection!(mcts_UCT_1, init_board, 1, 1, 100)
   23.400 μs (275 allocations: 10.11 KiB)
(0x0003000000010002, Any[(0x0100010000000000, 0x00, 0.0f0), (0x0011000100000000, 0x01, 4.0f0), (0x1012000000000000, 0x00, 8.0f0), (0x0022000000000020, 0x00, 8.0f0), (0x0003000000010002, 0x00, -1.0f0), (0x0003000000010002, 0x00, -1.0f0), (0x0003000000010002, 0x00, -1.0f0), (0x0003000000010002, 0x00, -1.0f0), (0x0003000000010002, 0x00, -1.0f0), (0x0003000000010002, 0x00, -1.0f0)  …  (0x0003000000010002, 0x00, -1.0f0), (0x0003000000010002, 0x00, -1.0f0), (0x0003000000010002, 0x00, -1.0f0), (0x0003000000010002, 0x00, -1.0f0), (0x0003000000010002, 0x00, -1.0f0), (0x0003000000010002, 0x00, -1.0f0), (0x0003000000010002, 0x00, -1.0f0), (0x0003000000010002, 0x00, -1.0f0), (0x0003000000010002, 0x00, -1.0f0), (0x0003000000010002, 0x00, -1.0f0)])=#

#since the timing for this procedure is much longer than the ~1 microsecond time to do a reasonable rollout, we won't benefit from parallelism since the simulation step isn't the bottleneck.  We can make it the bottleneck though if we average ~100 rollouts

#let's see if the policy on the board10 state makes sense
mcts_UCT_1(board10)

function play_mcts_UCT_game(rolloutestimatedepth, numrollouts, mctsdepth, c, numsims)
    π = init_MCTSPar(mdp_2048, init_board, s -> mean(run_random_2048_rollout(s, rolloutestimatedepth)[1] for _ in 1:numrollouts), 1, 1, UCT(), d = mctsdepth, c = c, m = numsims)
    play_game(s -> π(s))
end

#higher MCTS depth makes it more likely that only leaf states will be simulated
play_mcts_UCT_game(10000, 1, 100, 50., 100)
#=
julia> @time play_mcts_UCT_game(10000, 1, 100, 50., 100)
  9.729531 seconds (125.07 M allocations: 4.004 GiB, 4.79% gc time)
(15736.0f0, 10, 857)
=#

mcts_treeP_1 = init_MCTSPar(mdp_2048, init_board, s -> run_random_2048_rollout(s, 10000)[1], 1, 20, TreeP(), d = 10, c = 10., m = 100)
mcts_treeP_1(init_board)
mcts_treeP_1(board10)

function play_mcts_par_game(rolloutestimatedepth, numrollouts, mctsdepth, c, numsims, ntrees, algo, simulators=20)
    π = init_MCTSPar(mdp_2048, init_board, s -> mean(run_random_2048_rollout(s, rolloutestimatedepth)[1] for _ in 1:numrollouts), ntrees, simulators, algo, d = mctsdepth, c = c, m = numsims)
    play_game(s -> π(s))
end

play_mcts_par_game(1000, 1, 100, 50., 100, 1, TreeP())

#what can we make the evaluation more complex without losing time?
play_mcts_par_game(1000, 10, 100, 50., 100, 1, TreeP())
play_mcts_par_game(1000, 100, 100, 50., 200, 1, TreeP())

mcts_rootP_1 = init_MCTSPar(mdp_2048, init_board, s -> run_random_2048_rollout(s, 10000)[1], 1, 20, RootP(), d = 1000, c = 100., m = 1000)
mcts_rootP_1(init_board)
mcts_rootP_1(board10)

play_mcts_par_game(1000, 100, 100, 50., 200, 20, RootP())

mcts_WU_UCT_1 = init_MCTSPar(mdp_2048, init_board, s -> run_random_2048_rollout(s, 10000)[1], 1, 20, WU_UCT(), d = 1000, c = 100., m = 1000)
mcts_WU_UCT_1(init_board)
mcts_WU_UCT_1(board10)

play_mcts_par_game(1000, 100, 100, 50., 100, 1, WU_UCT())

mcts_BU_UCT_1 = init_MCTSPar(mdp_2048, init_board, s -> run_random_2048_rollout(s, 10000)[1], 1, 20, BU_UCT(0.5), d = 1000, c = 100., m = 1000)
mcts_BU_UCT_1(init_board)
mcts_BU_UCT_1(board10)

play_mcts_par_game(1000, 100, 100, 50., 100, 1, BU_UCT(0.5))
