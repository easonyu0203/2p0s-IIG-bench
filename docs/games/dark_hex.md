# Dark Hex 3

`nashbench.make("dark_hex3")` and `nashbench.make("dark_hex3_abrupt")`
replicate OpenSpiel's
[`dark_hex`](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/games/dark_hex/dark_hex.h)
with its default 3x3 board and `obstype=reveal-nothing`, and with
`gameversion=cdh` (classical) and `gameversion=adh` (abrupt) respectively.

Hex, but you can't see your opponent's stones. Seat 0 plays black, moves
first, and wins by connecting the top and bottom rows. Seat 1 plays white
and wins by connecting the left and right columns. Hex has no draws. If you
try a cell that your opponent occupies, you see their stone there. Then, in
the classical version, you try again. In the abrupt version, your turn ends.
You aren't told how many times your opponent has tried.

## Actions

Action `i` tries cell `i`. Cells are numbered row by row, and each row is
shifted right by half a cell:

```text
0 1 2
 3 4 5
  6 7 8
```

Cell 4, for example, neighbors cells 1, 2, 3, 5, 6, and 7. A cell is legal if
you haven't seen a stone on it.

## Observation

164 dimensions:

| Dimensions | Meaning |
| --- | --- |
| 0–1 | Seat, one-hot. |
| 2–82 | For each cell `i`, dimensions `2 + 9i` to `10 + 9i` one-hot encode OpenSpiel's hex cell state. Before the game ends, only offset 3 (white), 4 (no stone seen), and 5 (black) occur. |
| 83–163 | Your tries in order: nine slots, each one-hot over the nine cells. Unused slots are zeros. |

```{note}
OpenSpiel's `dark_hex` information-state *string* also contains the total
number of moves, which reveals how many times your opponent has tried.
OpenSpiel's information-state *tensor* doesn't, and nashbench follows the
tensor. As a result, several OpenSpiel strings can map to one nashbench
observation. Tools that key information states by these strings, such as
OpenSpiel's tabular best response, give the best responder this extra
information.
```

## Rewards

1 for a win and −1 for a loss.
