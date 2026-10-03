# Limit poker

`nashbench.make("universal_poker")` replicates OpenSpiel's
[`universal_poker`](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/games/universal_poker/universal_poker.h)
with limit betting over four rounds:
`universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=1 1,raiseSize=2 2 4 4,firstPlayer=1 1 1 1,maxRaises=2 2 2 2,numSuits=4,numRanks=3,numHoleCards=1,numBoardCards=0 1 1 1)`.
To change the deck, the rounds, or the raises, pass `num_ranks`,
`raise_sizes` (one per round, for one to four rounds), or `max_raises`.

The deck has four suits of `num_ranks` ranks: 2, 3, and 4 by default. Each
seat antes 1 chip and is dealt one private card. Each round after the first
reveals a public card. Seat 0 acts first in every round. A raise adds 2
chips in rounds 1 and 2 and 4 chips in rounds 3 and 4, and each round allows
at most two raises.

At a showdown, a seat's hand is its private card and the public cards. Four
of a kind beats three of a kind, which beats two pairs, then a pair, then no
pair. Within these, the higher ranks win, those of the larger groups first.
Equal hands split the pot.

> [!NOTE]
> OpenSpiel's information states show the public cards as a set, which
> forgets the round that revealed each. With more than one public card, they
> lack perfect recall. nashbench's observations append the public card of
> each round.

## Actions

| ID | Action | Legal when |
| --- | --- | --- |
| 0 | Fold | Facing a raise. |
| 1 | Call, or check if not facing a raise | Always. |
| 2 | Raise | Fewer than `max_raises` raises in this round. |

## Observation

`2 + (r + 1)d + 3(3r + 22)` dimensions for `d` cards and `r` rounds. Card `c`
has rank `c // 4` and suit `c % 4`. For the default, 164 dimensions:

| Dimensions | Meaning |
| --- | --- |
| 0–1 | Seat, one-hot. |
| 2–13 | Own card, one-hot over the 12 cards. |
| 14–25 | Public cards revealed so far. |
| 26–93 | OpenSpiel's action sequence: 34 slots in order, each one-hot over (call, raise). Each deal takes a slot with zeros: two before round 1 and one before each later round. Folds and empty slots are zeros. |
| 94–127 | Zeros, for OpenSpiel's raise sizes, which are 0 in limit games. |
| 128–163 | Public card of rounds 2, 3, and 4, each one-hot over the 12 cards. Zeros until revealed. |

## Rewards

Chips won or lost, from −25 to 25 by default.
