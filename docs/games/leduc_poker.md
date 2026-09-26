# Leduc poker

`nashbench.make("leduc_poker")` replicates OpenSpiel's
[`leduc_poker`](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/games/leduc_poker/leduc_poker.h).

The deck has six cards: J, Q, and K in two suits. Each seat antes 1 chip and
is dealt one private card. There are two betting rounds, and seat 0 acts
first in each. After the first round, a public card is revealed. A raise
adds 2 chips in round 1 and 4 chips in round 2, and each round allows at most
two raises.

At a showdown, a private card that pairs with the public card wins.
Otherwise, the higher private card wins, and equal ranks split the pot.

## Actions

| ID | Action | Legal when |
| --- | --- | --- |
| 0 | Fold | Facing a raise. |
| 1 | Call, or check if not facing a raise | Always. |
| 2 | Raise | Fewer than two raises in this round. |

## Observation

30 dimensions. Card `c` has rank `c // 2` (J, Q, K) and suit `c % 2`.

| Dimensions | Meaning |
| --- | --- |
| 0–1 | Seat, one-hot. |
| 2–7 | Own card, one-hot over the six cards. |
| 8–13 | Public card, one-hot. Zeros in round 1. |
| 14–21 | Round 1 actions: four slots in order, each one-hot over (call, raise). Folds and empty slots are zeros. |
| 22–29 | Round 2 actions, encoded like round 1. |

## Rewards

Chips won or lost, from −13 to 13.
