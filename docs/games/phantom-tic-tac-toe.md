# Phantom Tic-Tac-Toe

`nashbench.make("phantom_ttt")` and `nashbench.make("phantom_ttt_abrupt")`
replicate OpenSpiel's
[`phantom_ttt`](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/games/phantom_ttt/phantom_ttt.h)
with its default `obstype=reveal-nothing`, and with `gameversion=classical`
and `gameversion=abrupt` respectively.

Tic-Tac-Toe, but you can't see your opponent's marks. Seat 0 plays x and
moves first. If you try a cell that your opponent occupies, you see their
mark there. Then, in the classical version, you try again. In the abrupt
version, your turn ends. You aren't told how many times your opponent has
tried.

## Actions

Action `i` tries cell `i`. Cells are numbered row by row:

```text
0 1 2
3 4 5
6 7 8
```

A cell is legal if you haven't seen a mark on it.

## Observation

110 dimensions:

| Dimensions | Meaning |
| --- | --- |
| 0–1 | Seat, one-hot. |
| 2–10 | Cells where you haven't seen a mark. |
| 11–19 | Cells where you have seen o, the mark of seat 1. |
| 20–28 | Cells where you have seen x, the mark of seat 0. |
| 29–109 | Your tries in order: nine slots, each one-hot over the nine cells. Unused slots are zeros. |

## Rewards

1 for a win, −1 for a loss, and 0 for a draw.
