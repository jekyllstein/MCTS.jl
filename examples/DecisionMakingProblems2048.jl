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

@benchmark random_2048_policy($init_board)
#=
BenchmarkTools.Trial: 10000 samples with 999 evaluations.
 Range (min … max):  7.107 ns … 105.205 ns  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     7.808 ns               ┊ GC (median):    0.00%
 Time  (mean ± σ):   8.255 ns ±   2.284 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

     ▅█▄▃▁▂▃ ▂▁       ▁   ▃▃▂                                 ▁
  ▇▁▁███████▆███▆▅▄▄▃▆█▁▃▃████▁▄▁▁▁▁▁▃▅▁▄▃▁▃▁▁▃▃▄▅▄▅▁▃▆▄▆▅▅▄▅ █
  7.11 ns      Histogram: log(frequency) by time      15.6 ns <

 Memory estimate: 0 bytes, allocs estimate: 0.
=#


#if we use this policy for a rollout of 10 moves then we can see the reward accumulated after that many moves.  This will be important later for use in a value function estimate.
run_random_2048_rollout(10, init_board) #21.64

#let's also benchmark random rollouts to see how the time scales with length
rollout_times = [@benchmark run_random_2048_rollout($n, $init_board) for n in 2 .^(0:7)]
#=
8-element Vector{BenchmarkTools.Trial}:
 528.571 ns
 1.000 μs
 1.950 μs
 4.014 μs
 8.100 μs
 16.000 μs
 32.500 μs
 62.100 μs
 
 So we see the time roughly doubling each time the rollout length is doubled as expected
 =#

#now let's create some MCTS policies for 2048 that differ by their value function estimate
#the simplest mcts policy will just use the reward of the current board state as the value function estimate
mcts_board_score = create_mcts_policy(s -> DecisionMakingProblems.score_board(s))

#we can directly use the policy to evaluate the move it would suggest for the initial board just like we did for the random policy
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
@benchmark mcts_board_score($init_board) 
#=
BenchmarkTools.Trial: 1386 samples with 1 evaluation.
 Range (min … max):  2.943 ms … 48.442 ms  ┊ GC (min … max): 0.00% … 92.01%
 Time  (median):     3.441 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   3.600 ms ±  2.102 ms  ┊ GC (mean ± σ):  2.69% ±  4.31%

                 ▂█▆▃
  ▂▂▃▆▅▅▃▃▃▃▃▂▃▃▆██████▆▇▆▆▅▄▅▄▄▄▄▄▄▃▃▃▃▃▄▃▃▃▂▃▂▂▂▂▂▂▂▂▂▁▁▁▂ ▃
  2.94 ms        Histogram: frequency by time        4.47 ms <

 Memory estimate: 949.08 KiB, allocs estimate: 38480.
=#

#we can explore how adjusting the number of simulations affects the runtime
@benchmark create_mcts_policy($(s -> DecisionMakingProblems.score_board(s)), m = 1000)($init_board)
#=
BenchmarkTools.Trial: 471 samples with 1 evaluation.
 Range (min … max):   8.577 ms … 53.279 ms  ┊ GC (min … max): 0.00% … 81.45%
 Time  (median):     10.314 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   10.620 ms ±  3.937 ms  ┊ GC (mean ± σ):  3.36% ±  7.38%

               ▃▂▃▁▇█▄▁▂▁
  ▃▃▅▇▆▄▄▆█▄▃▃▆███████████▆▅▆▃▃▄▃▂▁▂▁▁▁▂▁▂▁▁▃▁▁▁▁▁▁▁▁▁▂▂▁▁▁▁▃ ▃
  8.58 ms         Histogram: frequency by time        14.4 ms <

 Memory estimate: 3.93 MiB, allocs estimate: 120246.
 =#
@benchmark create_mcts_policy($(s -> DecisionMakingProblems.score_board(s)), m = 10000)($init_board)
#=
BenchmarkTools.Trial: 32 samples with 1 evaluation.
 Range (min … max):  137.028 ms … 189.785 ms  ┊ GC (min … max): 0.00% … 22.60%
 Time  (median):     156.600 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   158.510 ms ±  11.874 ms  ┊ GC (mean ± σ):  2.53% ±  6.77%

                      ▆█ ▃▁
  ▄▁▄▁▇▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁██▇██▁▁▁▁▄▇▁▁▄▁▁▁▁▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄▁▄▁▄ ▁
  137 ms           Histogram: frequency by time          190 ms <

 Memory estimate: 47.61 MiB, allocs estimate: 1797372.

 So at 10000 simulations we see garbage collection having a noticeable effect on the runtime

 =#

 #we can get some more fine grained statistics using the following
function get_mcts_move_times_vs_simulations(U::Function; mvec = 2 .^(0:14))
  outputs = Vector{BenchmarkTools.Trial}(undef, length(mvec))
  @threads for i in eachindex(outputs)
    outputs[i] = @benchmark $create_mcts_policy($U, m = $mvec[$i])($init_board)
  end
  median_times = [median(a.times) for a in outputs]
  return mvec, median_times, outputs
end

function get_mcts_move_times_vs_depth(U::Function; m = 100, dvec = 2 .^(0:7))
  outputs = Vector{BenchmarkTools.Trial}(undef, length(dvec))
  @threads for i in eachindex(outputs)
    outputs[i] = @benchmark $create_mcts_policy($U, m = $m, d = $dvec[$i])($init_board)
  end
  median_times = [median(a.times) for a in outputs]
  return dvec, median_times, outputs
end


#using these functions we can get statistics on the score based utility function estimate policy
mcts_score_move_simulation_stats = get_mcts_move_times_vs_simulations(s -> score_board(s))
mcts_score_move_depth_stats = get_mcts_move_times_vs_depth(s -> score_board(s))

#now we can time the MCTS policy evaluation on a different version of the policy that uses a 10 step random rollout for the value function estimate 
mcts_rollout = create_mcts_policy(s -> run_random_2048_rollout(10, s))
mcts_rollout(init_board) #0x03
@benchmark mcts_rollout($init_board)
#=
BenchmarkTools.Trial: 1192 samples with 1 evaluation.
 Range (min … max):  3.324 ms … 49.355 ms  ┊ GC (min … max): 0.00% … 92.33%
 Time  (median):     4.027 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   4.189 ms ±  2.365 ms  ┊ GC (mean ± σ):  2.48% ±  4.58%

               ▁▃▂▅▃▃█▆▅▃▄▂▆▃ ▁
  ▃▂▄▄▅▆▆▇▆▆▅▆███████████████▇█▆█▆▆▆▅▅▅▅▃▃▄▃▃▃▃▃▂▂▁▁▁▂▁▁▂▂▁▂ ▄
  3.32 ms        Histogram: frequency by time        5.29 ms <

 Memory estimate: 1.14 MiB, allocs estimate: 47297.
=#

#not a huge difference in time for the default 100 simulations, but let's see how it scales up
@benchmark create_mcts_policy($(s -> run_random_2048_rollout(10, s)), m = 1000)($init_board)
#=
BenchmarkTools.Trial: 268 samples with 1 evaluation.
 Range (min … max):  14.851 ms … 60.537 ms  ┊ GC (min … max): 0.00% … 74.04%
 Time  (median):     18.244 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   18.663 ms ±  5.331 ms  ┊ GC (mean ± σ):  3.54% ±  9.00%

  ▅▂ ▅█▇
  ███████▄▄▆▁▄▁▅▄▁▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄ ▅
  14.9 ms      Histogram: log(frequency) by time      59.2 ms <

 Memory estimate: 7.30 MiB, allocs estimate: 261066.
=#

@benchmark create_mcts_policy($(s -> run_random_2048_rollout(10, s)), m = 10000)($init_board)
#=
BenchmarkTools.Trial: 21 samples with 1 evaluation.
 Range (min … max):  223.240 ms … 266.375 ms  ┊ GC (min … max): 0.00% … 17.45%
 Time  (median):     238.618 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   242.209 ms ±  11.805 ms  ┊ GC (mean ± σ):  3.67% ±  7.15%

                 ▃    █  ▃                                    ▃
  ▇▁▁▇▁▁▁▁▁▁▁▁▇▁▁█▁▁▇▇█▇▁█▁▁▁▇▇▁▁▁▁▇▇▁▁▁▁▁▁▁▁▁▁▁▁▁▁▇▁▁▇▁▁▁▁▁▁▁█ ▁
  223 ms           Histogram: frequency by time          266 ms <

 Memory estimate: 80.77 MiB, allocs estimate: 3184134.
 =#

 #so this rollout method is ~1.5-2x slower if the number of simulations scale up, but what is the impact on how well it plays the game?  

#we can use the play_game function to play a game to completion for a given policy.  Let's try it with the random policy
(random_score, random_rank, random_move_count) = play_game(random_2048_policy) #(1400.0f0, 7, 138) where 1400 is the score, 7 is the max rank which corresponds to a tile of 128, and 138 is the number of moves before the game ended

#using this function we can get statistics about games played with the random policy
random_games = [play_game(random_2048_policy) for _ in 1:1000]

randomgame_stats = getgamestats(random_games)
#=
3-element Vector{StatsBase.SummaryStats}:
 Summary Stats:
Length:         1000
Missing Count:  0
Mean:           1003.624023
Minimum:        4.000000
1st Quartile:   620.000000
Median:         960.000000
3rd Quartile:   1280.000000
Maximum:        3092.000000

 Summary Stats:
Length:         1000
Missing Count:  0
Mean:           6.473000
Minimum:        2.000000
1st Quartile:   6.000000
Median:         7.000000
3rd Quartile:   7.000000
Maximum:        8.000000

 Summary Stats:
Length:         1000
Missing Count:  0
Mean:           105.153000
Minimum:        3.000000
1st Quartile:   80.000000
Median:         103.000000
3rd Quartile:   126.000000
Maximum:        242.000000
=#
#so now we have some baseline stats to work with for all the relevent game characteristics

#also we can benchmark the play game function to see how long the random policy takes
@benchmark play_game($random_2048_policy)
#=
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):   1.600 μs … 139.600 μs  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     15.200 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   16.188 μs ±   6.818 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

              ▁ ▄▆▇▆▄▆▆▆▆▅█▅▆▃▅▃▁
  ▁▁▁▁▁▁▂▂▃▃▅▆████████████████████▇▇▆▆▄▅▄▅▄▄▃▃▃▃▂▃▂▂▂▂▂▁▁▂▁▂▁▁ ▄
  1.6 μs          Histogram: frequency by time         36.3 μs <

 Memory estimate: 336 bytes, allocs estimate: 20.
=#

#now let's try playing a game with the MCTS policy that uses the board score as the value function estimate and uses 100 simulations.
mcts_score_100_policy = create_mcts_policy(s -> score_board(s))
mcts_score_100_game = play_game(s -> mcts_score_100_policy(s)) #(7008.0f0, 9, 451)

@benchmark play_game($(s -> mcts_score_100_policy(s)))
#=
BenchmarkTools.Trial: 7 samples with 1 evaluation.
 Range (min … max):  472.397 ms …    1.258 s  ┊ GC (min … max): 0.00% … 2.15%
 Time  (median):     756.019 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   804.385 ms ± 264.823 ms  ┊ GC (mean ± σ):  2.47% ± 3.61%

  █         █  █        █          █     █                    █  
  █▁▁▁▁▁▁▁▁▁█▁▁█▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  472 ms           Histogram: frequency by time          1.26 s <

 Memory estimate: 120.07 MiB, allocs estimate: 4882559.
 =#

 #and now we can get statisics on the basic mcts policy
mcts_score_100_games = [play_game(b -> create_mcts_policy(s -> score_board(s))(b)) for _ in 1:100]
mcts_score_100_stats = getgamestats(mcts_score_100_games)
#=
3-element Vector{StatsBase.SummaryStats}:
 Summary Stats:
Length:         100
Missing Count:  0
Mean:           7069.879883
Minimum:        1296.000000
1st Quartile:   5510.000000
Median:         6876.000000
3rd Quartile:   7187.000000
Maximum:        15100.000000

 Summary Stats:
Length:         100
Missing Count:  0
Mean:           9.050000
Minimum:        7.000000
1st Quartile:   9.000000
Median:         9.000000
3rd Quartile:   9.000000
Maximum:        10.000000

 Summary Stats:
Length:         100
Missing Count:  0
Mean:           437.770000
Minimum:        124.000000
1st Quartile:   363.250000
Median:         435.500000
3rd Quartile:   465.250000
Maximum:        816.000000
=#

#let's see if performance improves with 10x the depth
mcts_score_100m_100d_games = Vector{Any}(undef, 100)
@threads for i in 1:100
  mcts_score_100m_100d_games[i] = play_game(b -> create_mcts_policy(s -> score_board(s), d = 100)(b))
end
mcts_score_100m_100d_stats = getgamestats(mcts_score_100m_100d_games)
#=
3-element Vector{StatsBase.SummaryStats}:
 Summary Stats:
Length:         100
Missing Count:  0
Mean:           5923.600098
Minimum:        1368.000000
1st Quartile:   3388.000000
Median:         6206.000000
3rd Quartile:   6938.000000
Maximum:        13004.000000

 Summary Stats:
Length:         100
Missing Count:  0
Mean:           8.760000
Minimum:        7.000000
1st Quartile:   8.000000
Median:         9.000000
3rd Quartile:   9.000000
Maximum:        10.000000

 Summary Stats:
Length:         100
Missing Count:  0
Mean:           382.250000
Minimum:        136.000000
1st Quartile:   269.000000
Median:         398.500000
3rd Quartile:   442.750000
Maximum:        723.000000
=#

#let's see if performance improves with 10x the simulations
mcts_score_1000m_10d_games = Vector{Any}(undef, 100)
@threads for i in 1:100
  mcts_score_1000m_10d_games[i] = play_game(b -> create_mcts_policy(s -> score_board(s), m = 1000)(b))
end
mcts_score_1000m_10d_stats = getgamestats(mcts_score_1000m_10d_games)
#=
3-element Vector{StatsBase.SummaryStats}:
 Summary Stats:
Length:         100
Missing Count:  0
Mean:           11791.440430
Minimum:        5148.000000
1st Quartile:   7014.000000
Median:         11892.000000
3rd Quartile:   15020.000000
Maximum:        26396.000000

 Summary Stats:
Length:         100
Missing Count:  0
Mean:           9.710000
Minimum:        9.000000
1st Quartile:   9.000000
Median:         10.000000
3rd Quartile:   10.000000
Maximum:        11.000000

 Summary Stats:
Length:         100
Missing Count:  0
Mean:           657.350000
Minimum:        319.000000
1st Quartile:   449.750000
Median:         658.000000
3rd Quartile:   805.250000
Maximum:        1270.000000
=#

#now let's compare to an mcts policy that uses a 10 step random rollout instead of the board score for the value function estimate
mcts_rollout_100_policy = create_mcts_policy(s -> run_random_2048_rollout(10, s))
mcts_rollout_100_game = play_game(s -> mcts_rollout_100_policy(s)) #(35704, 11, 1741)
mcts_rollout_100_games = [play_game(b -> create_mcts_policy(s -> run_random_2048_rollout(10, s))(b)) for _ in 1:100]
mcts_rollout_100_game_stats = getgamestats(mcts_rollout_100_games)
#=
3-element Vector{StatsBase.SummaryStats}:
 Summary Stats:
Length:         100
Missing Count:  0
Mean:           25563.839844
Minimum:        6992.000000
1st Quartile:   16011.000000
Median:         26776.000000
3rd Quartile:   32105.000000
Maximum:        59116.000000

 Summary Stats:
Length:         100
Missing Count:  0
Mean:           10.640000
Minimum:        9.000000
1st Quartile:   10.000000
Median:         11.000000
3rd Quartile:   11.000000
Maximum:        12.000000

 Summary Stats:
Length:         100
Missing Count:  0
Mean:           1257.940000
Minimum:        449.000000
1st Quartile:   888.250000
Median:         1302.500000
3rd Quartile:   1541.500000
Maximum:        2535.000000
=#

#even with 100 simulations this seems to give much better results, with just the first game already reaching the max rank.  What if we do it with 1000 simulations instead
mcts_rollout_1000_policy = create_mcts_policy(s -> run_random_2048_rollout(10, s), m = 1000)
mcts_rollout_1000_game = play_game(s -> mcts_rollout_1000_policy(s)) #(27056.0f0, 11, 1329)

#and with 10000
# mcts_rollout_10000_policy = create_mcts_policy(s -> run_random_2048_rollout(10, s), m = 10000)
# mcts_rollout_10000_game = play_game(s -> mcts_rollout_10000_policy(s))  #(35716.0f0, 11, 1743)

 #What if we only use 10 simulations
mcts_rollout_10_policy = create_mcts_policy(s -> run_random_2048_rollout(10, s), m = 10)
mcts_rollout_10_game = play_game(s -> mcts_rollout_10_policy(s)) #(6272.0f0, 9, 402)
@benchmark play_game($(s -> mcts_rollout_10_policy(s)))
#=
BenchmarkTools.Trial: 54 samples with 1 evaluation.
 Range (min … max):  13.565 ms … 226.403 ms  ┊ GC (min … max): 0.00% … 19.36%
 Time  (median):     90.024 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   94.552 ms ±  51.557 ms  ┊ GC (mean ± σ):  2.29% ±  5.39%

  █     ▃      █    ▃▃  ▃▃▃▃█ ▃▃        █▃                  ▃
  █▇▁▇▁▁█▇▁▇▇▇▇█▇▇▇▁██▇▇█████▁██▇▁▁▁▁▁▇▇██▁▇▁▁▇▁▁▁▁▁▁▇▁▁▁▁▁▁█▇ ▁
  13.6 ms         Histogram: frequency by time          209 ms <

 Memory estimate: 3.75 MiB, allocs estimate: 154297.
=#
mcts_rollout_10_games = [play_game(b -> create_mcts_policy(s -> run_random_2048_rollout(10, s), m = 10)(b)) for _ in 1:100]
mcts_rollout_10_game_stats = getgamestats(mcts_rollout_10_games)
#=
3-element Vector{StatsBase.SummaryStats}:
 Summary Stats:
Length:         100
Missing Count:  0
Mean:           8728.000000
Minimum:        2096.000000
1st Quartile:   6190.000000
Median:         7044.000000
3rd Quartile:   12270.000000
Maximum:        15896.000000

 Summary Stats:
Length:         100
Missing Count:  0
Mean:           9.260000
Minimum:        8.000000
1st Quartile:   9.000000
Median:         9.000000
3rd Quartile:   10.000000
Maximum:        10.000000

 Summary Stats:
Length:         100
Missing Count:  0
Mean:           513.460000
Minimum:        159.000000
1st Quartile:   393.250000
Median:         452.500000
3rd Quartile:   692.750000
Maximum:        885.000000
=#

#now let's see how much the policy improves using a 100 step random rollout instead
mcts_rollout100_100m_policy = create_mcts_policy(s -> run_random_2048_rollout(100, s))
mcts_rollout100_100m_game = play_game(s -> mcts_rollout100_100m_policy(s)) #(14668.0f0, 10, 769)
#just from one game we can see it is much worse, what if instead we do the average reward of 100 10x rollouts?

#this is an example of leaf parallelism where we make the value function estimate parallel.  It doesn't work very well
function parallel_rollout(d, s; n = 100)
  rs = Vector{Float64}(undef, n)
  @threads for i in 1:n
    rs[i] = run_random_2048_rollout(d, s)
  end
  return mean(rs)
end

mcts_parallelrollout10_100m_policy = create_mcts_policy(s -> parallel_rollout(10, s))
mcts_parallelrollout10_100m_game = play_game(s -> mcts_parallelrollout10_100m_policy(s)) #(32060.0f0, 11, 1538)

mcts_parallelrollout10_100m_policy2 = create_mcts_policy(s -> parallel_rollout(10, s, n = 10))
mcts_parallelrollout10_100m_game2 = play_game(s -> mcts_parallelrollout10_100m_policy2(s)) #(70232.0f0, 12, 2994)

#let's try the tree parallelism version instead
mcts_treepar_score_policy = create_treepar_mcts_policy(s -> score_board(s))
# currently crashing
# tree_par_test_game = play_game(s -> mcts_treepar_score_policy(s))