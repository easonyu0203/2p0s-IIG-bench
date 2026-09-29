# Liar's Dice

`nashbench.make("liars_dice")` replicates OpenSpiel's
[`liars_dice`](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/games/liars_dice/liars_dice.h)
with two five-sided dice per seat: `liars_dice(numdice=2,dice_sides=5)`. For
other sizes, pass `numdice` and `dice_sides`, for example OpenSpiel's
default, `nashbench.make("liars_dice", numdice=1, dice_sides=6)`. On a 24 GB
GPU, exact evaluation fits up to about 50 million information sets: the
default has 15.7 million, and two six-sided dice would have 352 million.

Each seat rolls its dice and sees only its own. Seats then take turns,
starting with seat 0. A seat either bids that at least `q` of all dice show
face `f`, or calls the last bid a lie. A bid must be higher than the last
one: more dice, or as many dice with a higher face. The highest face is wild
and counts as every face. After a call, the bidder wins if its bid holds, and
the caller wins otherwise.

## Actions

Action `(q - 1) * dice_sides + f - 1` bids that at least `q` dice show face
`f`. It's legal if it's higher than the last bid. The last action,
`2 * numdice * dice_sides` (20 by default), calls liar. It's legal after the
first bid.

## Observation

`3 * numdice * dice_sides + 3` dimensions. For the default, 33 dimensions:

| Dimensions | Meaning |
| --- | --- |
| 0–1 | Seat, one-hot. |
| 2–11 | Own dice in increasing order, each one-hot over the five faces. |
| 12–31 | Bids made so far. |
| 32 | Liar called. Always zero when a seat acts. |

## Rewards

1 for the winner and −1 for the loser.
