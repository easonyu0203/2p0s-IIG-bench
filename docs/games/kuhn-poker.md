# Kuhn poker

`nashbench.make("kuhn_poker")` replicates OpenSpiel's
[`kuhn_poker`](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/games/kuhn_poker/kuhn_poker.h).

Each seat antes 1 chip and is dealt one card from a deck of three: J, Q, and
K. Seat 0 acts first. Players can pass or bet 1 chip. If a player bets, the
other player either calls or folds. At a showdown, the higher card wins.

## Actions

| ID | Action |
| --- | --- |
| 0 | Pass: check, or fold if facing a bet. |
| 1 | Bet: bet, or call if facing a bet. |

Both actions are always legal.

## Observation

11 dimensions:

| Dimensions | Meaning |
| --- | --- |
| 0–1 | Seat, one-hot. |
| 2–4 | Own card, one-hot: J, Q, K. |
| 5–10 | Up to three betting actions in order, each one-hot over (pass, bet). Actions not yet taken are zeros. |

## Rewards

Chips won or lost: ±1, or ±2 if a bet is called.
