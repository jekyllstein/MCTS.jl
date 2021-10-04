# Setup
Open a julia 1.6 REPL in the main directory with the --project option.  Then enter package mode by typing `]` and type `instantiate` to install dependencies.  After this hit `backspace` to enter the normal julia repl and the module will be accessible by typing `using MCTS`

# Run example
Open a julia 1.6 REPL in the MCTS environment using the --project option.  If the working directory is the main MCTS folder, you can run the code in the 'game2048.jl' file in the examples folder by typing `include(joinpath(examples, "game2048.jl"))` which should print out a gameboard after a series of random moves.