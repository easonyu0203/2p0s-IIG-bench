"""The sequence form of Liar's Dice, from the regular structure of its tree.

Bids only increase, so the bids made are a bitmask `x`, and the seat to act
is the parity of its number of bids. `x // 2` numbers the bitmasks of one
parity, so a seat's information set, with dice of rank `r` among the
`num_dice` sorted dice a seat can roll, is numbered

    seat * num_per_seat + x // 2 * num_dice + r.

Information sets with `x // 2` in `[2**i // 2, 2**i)` follow bid `i`, or no
bid at seat 0's root. Their parents drop the last two bids, so they come
earlier.
"""

import dataclasses
import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np

from nashbench import sequence_form as sequence_form_lib


def sequence_form(game) -> sequence_form_lib.SequenceForm:
    """Returns the sequence form of a Liar's Dice game."""
    num_bids, sides = game.num_bids, game.dice_sides
    # The sorted dice a seat can roll, and their probabilities.
    rolls = itertools.product(range(sides), repeat=game.numdice)
    dice, count = np.unique(np.sort(list(rolls)), axis=0, return_counts=True)
    prob = count / count.sum()
    # The return of the bidder of each bid, by the dice of both seats and
    # weighted by chance; symmetric in the seats. Bid `b` holds if more than
    # `b // sides` dice show face `b % sides` or the wild face.
    quantity, face = np.divmod(np.arange(num_bids), sides)
    matches = ((dice == face[:, None, None]) | (dice == sides - 1)).sum(-1)
    holds = matches[:, :, None] + matches[:, None, :] > quantity[:, None, None]
    weight = np.where(holds, 1.0, -1.0) * np.outer(prob, prob)

    num_dice = len(dice)
    num_per_seat = num_dice << (num_bids - 1)
    parent, legal = _parent_and_legal(
        num_bids, num_dice, jnp.arange(2 * num_per_seat)
    )
    levels = tuple(
        (s * num_per_seat + start * num_dice, s * num_per_seat + end * num_dice)
        for start, end in (_range(i) for i in range(num_bids))
        for s in range(2)
    )
    return sequence_form_lib.SequenceForm(
        levels=levels,
        parent=parent,
        legal_action_mask=legal,
        observe=jax.jit(functools.partial(_observe, game, jnp.asarray(dice))),
        gradient=jax.jit(
            functools.partial(
                _gradient, num_dice, jnp.asarray(weight, jnp.float32)
            )
        ),
    )


def _describe(num_bids, num_dice, ids):
    """Returns the seat, bids made, and dice rank of information sets."""
    seat, local = jnp.divmod(ids, num_dice << (num_bids - 1))
    half, rank = jnp.divmod(local, num_dice)
    x = 2 * half + ((jax.lax.population_count(half) + seat) & 1)
    return seat, x, rank


def _range(i):
    """Returns the range of `x // 2` of information sets that follow bid `i`."""
    return 2**i >> 1, 2**i


def _highest(x):
    """Returns the highest bit set in `x`, or -1 if none."""
    return 31 - jax.lax.clz(x)


@functools.partial(jax.jit, static_argnums=(0, 1))
def _parent_and_legal(num_bids, num_dice, ids):
    """Returns the parent sequence and legal actions of information sets."""
    seat, x, rank = _describe(num_bids, num_dice, ids)
    # The seat's last bid is the second highest; the highest is the reply.
    below = x & ((1 << _highest(x)) - 1)
    last = _highest(below)
    before = below & ((1 << last) - 1)
    infoset = seat * (num_dice << (num_bids - 1)) + (before >> 1) * num_dice
    parent = (num_bids + 1) * (infoset + rank + 1) + last
    parent = jnp.where(jax.lax.population_count(x) < 2, seat, parent)
    action = jnp.arange(num_bids + 1)
    legal = (action > _highest(x)[:, None]) & (
        (action < num_bids) | (x[:, None] > 0)
    )
    return parent, legal


def _gradient(num_dice, weight, plan):
    """Returns the gradient of `plan`, summing over all terminal histories.

    In a terminal history, a seat calls liar at public state `x`, whose
    highest bid `i` the other seat made at `x - 2**i`. So the callers of bid
    `i`, with `x // 2` in `_range(i)`, match in order the first bidders of the
    other seat.
    """
    num_bids = weight.shape[0]
    # By seat, `x // 2`, dice, and action.
    plan = plan[num_bids + 1 :].reshape(2, -1, num_dice, num_bids + 1)
    dot = functools.partial(jnp.dot, precision=jax.lax.Precision.HIGHEST)
    bid_columns, liar_column = [], []
    for i in range(num_bids):
        start, end = _range(i)
        # By the seat of the caller. At bid 0, seat 0's callers are its root,
        # where calling is illegal, and so are their bidders' bids.
        callers = plan[:, start:end, :, num_bids]
        bidders = plan[::-1, : end - start, :, i]
        pad = ((0, 0), (0, plan.shape[1] - end + start), (0, 0))
        bid_columns.append(jnp.pad(dot(callers, weight[i])[::-1], pad))
        liar_column.append(-dot(bidders, weight[i]))
    gradient = jnp.stack([*bid_columns, jnp.concatenate(liar_column, 1)], -1)
    return jnp.pad(gradient.ravel(), (num_bids + 1, 0))


def _observe(game, dice, ids):
    """Returns observations of information sets, from representative states."""
    seat, x, rank = _describe(game.num_bids, len(dice), ids)
    states = jax.tree.map(
        lambda v: jnp.repeat(v[:1], ids.size, 0), game.initial_states()[0]
    )
    states = dataclasses.replace(
        states,
        # Only the seat's own dice matter.
        dice=jnp.repeat(dice[rank][:, None], 2, 1),
        bids=(x[:, None] >> jnp.arange(game.num_bids)) & 1 == 1,
    )
    return jax.vmap(game.observe)(states, seat)
