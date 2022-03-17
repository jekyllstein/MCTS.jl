include("DecisionMakingProblems2048_utilities.jl")

#create a starting board
init_board = initial_board()

U(s) = run_random_2048_rollout(10, s)

rollout_pol = init_MCTSPar(mdp_2048, init_board, U, 1, MCTSExperiments.UCT())

rollout_pol(init_board)