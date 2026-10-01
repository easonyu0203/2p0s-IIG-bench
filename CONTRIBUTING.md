# Contributing

Contributions are welcome, especially new games. For large changes, open an
issue first so we can agree on the design.

## Set up your environment

1.  Install [uv](https://docs.astral.sh/uv/).
1.  Clone the repository and install the package with its development tools:

    ```sh
    git clone https://github.com/easonyu0203/2p0s-IIG-bench
    cd 2p0s-IIG-bench
    uv sync
    ```

1.  Before you send a pull request, run the checks that CI runs:

    ```sh
    uv run ruff check
    uv run ruff format
    uv run pyright
    uv run pytest
    ```

If you use VS Code, install the recommended extensions when prompted. Ruff
and Pylance then flag problems as you type, files are formatted on save,
tests appear in the Testing view, and
<kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>B</kbd> runs all the checks.

## Code style

Code follows the
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html),
and docs follow the
[Google developer documentation style guide](https://developers.google.com/style).
Ruff enforces formatting, import order, and docstrings.

Docs are Markdown files that GitHub renders: [README.md](README.md) and
[docs/](docs). State each fact on one page and link to it from the others.
Docstrings are the API reference. In README.md, use absolute GitHub URLs,
because PyPI also renders it; elsewhere, use relative links.

Keep the code minimal. Every line should be needed to understand or run the
library. Comment only what the code can't say.

## Dependencies

nashbench supports JAX versions released in the past year. At each release,
raise the `jax` lower bound in `pyproject.toml` to the oldest version in that
window. CI tests both the lower bound and the latest versions.

## Add a game

nashbench games replicate OpenSpiel games: the same rules, action IDs,
returns, and information-state tensors. To add one:

1.  Find the game in OpenSpiel and read its `.cc` source.
1.  Create `src/nashbench/games/<name>.py` with a subclass of `core.Game`.
    Implement the rules for seats, where seat `i` is OpenSpiel's player `i`;
    `Game.reset` and `Game.step` handle players and the seat shuffle. For the
    simplest example, see `kuhn_poker.py`. If both seats move at once, take
    their moves in turn and hide seat 0's move until seat 1 moves, as in
    `goofspiel.py`.

    Games run in batches on accelerators, so write the rules with
    elementwise operations. For example, to set `x[i]` where `i` depends on
    the state, use `jnp.where(jnp.arange(x.size) == i, v, x)`: under
    `jax.vmap`, `x.at[i].set(v)` becomes a scatter, a separate GPU kernel.
1.  Make `observe` return a one-hot encoding of the seat followed by
    OpenSpiel's information-state tensor. If OpenSpiel's tensor already starts
    with the player, as in poker, don't repeat it.
1.  Register the game in `src/nashbench/games/__init__.py`.
1.  Add `docs/games/<name>.md`, which lists the actions and what each
    observation dimension means. Add the game to the tables in
    [docs/games/index.md](docs/games/index.md) and
    [docs/benchmarks.md](docs/benchmarks.md). [`benchmarks/measure.py`](benchmarks/measure.py)
    measures their numbers, except the number of terminal histories, which
    you count. Run it on an NVIDIA RTX A6000, the GPU for every measurement
    in the docs.
1.  Add the game to `GAMES` in `tests/openspiel_test.py`, wrapping games
    whose seats move at once in `turn_based_simultaneous_game(game=...)`. If
    the game has chance events, set them in `_replay`, and to check every
    history, list its deals in `DEALS`. The tests then compare the game with
    OpenSpiel at every node.

`exploitability` works for a new game without extra code if the game tree
fits in memory: `Game.sequence_form` enumerates it. For larger trees,
override `sequence_form`; `games/_phantom_tree.py` is an example that
traverses the tree on the accelerator at every evaluation.
