# Blotto

`nashbench.make("blotto")` replicates OpenSpiel's
[`blotto`](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/games/blotto/blotto.h)
with 10 coins and 5 fields: `blotto(coins=10,fields=5)`. To change them, pass
`coins` and `fields`, for example OpenSpiel's default,
`nashbench.make("blotto", fields=3)`.

Each seat splits its coins over the fields, at the same time. A seat wins a
field by putting more coins on it than the other seat, and the seat that wins
more fields wins the game. The game has a single decision per seat.

Seat 0 moves first, and seat 1 moves without seeing seat 0's move. See
[Simultaneous moves](../core-concepts.md#simultaneous-moves).

## Actions

Each action is a split of the coins, in lexicographic order of the coins on
each field: action 0 puts every coin on the last field, and the last action
puts every coin on the first field. 10 coins and 5 fields have 1,001 splits.
All actions are always legal.

## Observation

3 dimensions:

| Dimensions | Meaning |
| --- | --- |
| 0–1 | Seat, one-hot. |
| 2 | Whether the game has ended. Always zero when a seat acts. |

## Rewards

1 for winning more fields, −1 for fewer, and 0 for a tie.
