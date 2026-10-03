# Battleship

`nashbench.make("battleship")` replicates OpenSpiel's
[`battleship`](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/games/battleship/battleship.h)
with one ship of two cells on a 6x6 board and two shots per seat:
`battleship(board_height=6,board_width=6,ship_sizes=[2],ship_values=[1],num_shots=2,allow_repeated_shots=false)`.
To change the board, the ship, or the shots, pass `height`, `width`,
`ship_size`, or `num_shots`.

Each seat places its ship on its own board, horizontally or vertically, and
seat 0 places first. Seats don't see the other's ship. They then take turns
to shoot at a cell of the other's board, starting with seat 0, never at the
same cell twice. A seat sees where the other seat shoots, and whether its own
shots hit. The game ends when a ship sinks, with all its cells hit, or when
both seats have taken all their shots.

## Actions

For a board of `h` rows, `w` columns, and `n = h * w` cells, numbered row by
row:

| ID | Action | Legal when |
| --- | --- | --- |
| `c`, below `n` | Shoot at cell `c`. | Shooting, at a cell the seat hasn't shot. |
| `n + c` | Place the ship horizontally, from cell `c` to the right. | Placing, if the ship fits. |
| `2n + c` | Place the ship vertically, from cell `c` down. | Placing, if the ship fits, and it has more than one cell. |

## Observation

`9 + h + w + 2s(5 + h + w)` dimensions for `s` shots per seat. For the
default, 89 dimensions:

| Dimensions | Meaning |
| --- | --- |
| 0–1 | Seat, one-hot. |
| 2 | Whether the game has ended. Always zero when a seat acts. |
| 3–4 | Seat, one-hot, as OpenSpiel's tensor repeats it. |
| 5–6 | Seat to act, one-hot. |
| 7–8 | Own ship's direction, one-hot over (horizontal, vertical). Zeros before placing. |
| 9–14 | Row of its first cell, one-hot. |
| 15–20 | Column of its first cell, one-hot. |
| 21–88 | Every shot in order, starting with seat 0's: 17 dimensions each. The shooter, one-hot; the row and the column, one-hot; and for own shots, the outcome, one-hot over (water, hit, sunk). Shots not yet taken are zeros. |

## Rewards

1 for sinking the other seat's ship, −1 if your ship sinks, and 0 otherwise.
