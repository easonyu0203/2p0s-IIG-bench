# Benchmarks

Validation and measurements for the docs and the paper. They aren't part of
the nashbench package.

## Validation

`validate.py` checks nashbench's exploitability against independent
implementations. For each game and test policy, both get the same action
probabilities, and the script compares each seat's best-response value and
expected return. They must agree to within 10⁻⁹ times the game's largest
absolute return.

-   [OpenSpiel](https://github.com/google-deepmind/open_spiel) checks Kuhn
    and Leduc poker, and smaller Goofspiel, Liar's Dice and Oshi-Zumo. Their
    default sizes don't fit in OpenSpiel's memory.
-   [exp-a-spiel](https://github.com/gabrfarina/exp-a-spiel) checks the
    phantom Tic-Tac-Toe and Dark Hex games.

The test policies are uniform, random, deterministic, sparse (two actions
at most), a network, and a population that mixes different policies in each
seat. `common.py` defines them.

`traverse.py` checks the games themselves. `openspiel_test.py` compares random
histories of the larger games to keep the tests fast; `traverse.py` walks the
whole tree of every game at its default size together with OpenSpiel and
compares every history: whether the game is over, the returns, the seat to
act, its legal actions and both seats' observations. It also checks that
observations and OpenSpiel's information states determine each other and that
every deal has OpenSpiel's probability. The phantom games take hours on a
few dozen CPU cores (`--workers`).

## Measurements

`measure.py run` times exact evaluation and simulation on a GPU, and
`measure.py report` prints the results as tables. Each measurement answers
one question:

-   **`network`: how long does it take to evaluate a neural policy?** It
    calls `nashbench.evaluate` with a randomly initialized network, of two
    layers of 256 units or three of 512. The first call compiles. Later
    calls use new parameters each time, like checkpoints during training.
-   **`table`: how much of that time is the evaluator?** The same
    evaluation, with the network's probabilities computed beforehand.
-   **`exp_a_spiel`: how does nashbench compare with exp-a-spiel?** On the
    phantom Tic-Tac-Toe and Dark Hex games, exp-a-spiel evaluates the same
    probabilities on the CPU. It also times exp-a-spiel's full path:
    computing its observations, running the network, and evaluating.
-   **`population`: how does a population change the time?** Each seat
    mixes 1, 4, 16 or 64 networks.
-   **`simulation`: how fast do the games run?** Steps per second of 64,
    1,024 or 16,384 games at once, with random or network actions. Each
    step is one player's decision.
-   **`scaling`: how does that grow with the number of games?** The same,
    for 1 to 65,536 games at once on the games of the paper's core suite.
-   **`openspiel`: how fast is OpenSpiel, for reference?** OpenSpiel's
    implementation of each of those games, played from Python with random
    actions on one CPU thread, computing the acting seat's tensor at every
    decision.

Each measurement runs in a new process, so its first evaluation includes
compilation, as a user's does. Records also hold the time to build the
game's sequence form, which happens once per game, and the peak GPU memory.

`memory.py` records the peak host and GPU memory of one evaluation, in a new
process per game, for nashbench and for exp-a-spiel.

## Run

OpenSpiel comes with the development dependencies. For exp-a-spiel, install
it with Boost 1.81 or later, then run with `uv run --no-sync` so that uv
keeps it:

```sh
git clone https://github.com/gabrfarina/exp-a-spiel
git -C exp-a-spiel checkout 9412882c24e521fad7cbd96fdd68311d9bf6f203
CMAKE_PREFIX_PATH=$BOOST_PREFIX uv pip install ./exp-a-spiel
uv run --no-sync python benchmarks/validate.py
uv run --no-sync python benchmarks/measure.py run
uv run --no-sync python benchmarks/measure.py report
```

Both scripts append to an output file (`--out`) and skip what it already
has, and take `--only REGEX` to run some configurations.
